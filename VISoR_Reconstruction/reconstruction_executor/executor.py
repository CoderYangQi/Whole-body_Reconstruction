from VISoR_Reconstruction.reconstruction.sample_reconstruct import reconstruct_sample, reconstruct_image
from VISoR_Reconstruction.reconstruction.brain_reconstruct import calc_surface_height_map, extract_surface, align_surfaces, \
    process_transforms, create_brain, generate_brain_image, generate_projection, tile_images, generate_brain_projection
from VISoR_Brain.format.raw_data import RawData, load_raw_data
from VISoR_Brain.positioning.visor_sample import VISoRSample, load_visor_sample
from VISoR_Brain.positioning.visor_brain import VISoRBrain, load_visor_brain
from VISoR_Brain.utils.ome_tiff import write_ome_tiff
from VISoR_Brain.utils.simple_itk_utils import downsample_image
from VISoR_Reconstruction.reconstruction.brain_registration import generate_freesia_input
import os, json, sys, multiprocessing, time, gc, torch, traceback
import SimpleITK as sitk


target_definition = {
    'raw_data':    {
        'verify': os.path.isfile,
        'read_function': load_raw_data,
        'write_function': None
    },
    'reconstructed_slice':    {
        'verify': os.path.isfile,
        'read_function': load_visor_sample,
        'write_function': VISoRSample.save
    },
    'reconstructed_brain':    {
        'verify': os.path.exists,
        'read_function': load_visor_brain,
        'write_function': VISoRBrain.save
    },
    'image':    {
        'verify': os.path.isfile,
        'read_function': sitk.ReadImage,
        'write_function': sitk.WriteImage
    },
    'ome_tiff':    {
        'verify': os.path.isfile,
        'read_function': sitk.ReadImage,
        'write_function': write_ome_tiff
    },
    'image_sequence':    {
        'verify': os.path.isfile,
        'read_function': sitk.ReadImage,
        'write_function': sitk.WriteImage
    },
    'file':    {
        'verify': os.path.isfile,
        'read_function': lambda p: open(p, 'r').read(),
        'write_function': lambda s, p: open(p, 'w').write(s)
    },
    'null':    {
        'verify': lambda a: True,
        'read_function': lambda a: None,
        'write_function': lambda a, b: None
    }
}


task_definition = {
    (reconstruct_sample, 4, 0),
    (reconstruct_image, 32, 0),
    (calc_surface_height_map, 8, 1),
    (extract_surface, 8, 0),
    (downsample_image, 16, 0),
    (align_surfaces, 2, 0),
    (process_transforms, 4, 1),
    (create_brain, 2, 0),
    (generate_brain_image, 16, 0),
    (generate_projection, 16, 0),
    (generate_brain_projection, 8, 0),
    (generate_freesia_input, 1, 0),
    (tile_images, 4, 0)
}


task_definition = {
    f[0].__name__: {'function': f[0], 'resource': [f[1], f[2]]} for f in task_definition
}


resource_amount = [multiprocessing.cpu_count(), torch.cuda.device_count()]


class Target:
    def __init__(self, target_info: dict):
        self._data = None
        self.name = target_info['name']
        self.type = target_info['type']
        self.path = target_info['path']
        self.category = target_info['category']
        self.metadata = target_info['metadata']

        self.read_function = target_definition[self.type]['read_function']
        self.verify = target_definition[self.type]['verify']
        self.write_function = target_definition[self.type]['write_function']

    @property
    def data(self):
        return self._data

    def set_data(self, data):
        self._data = data

    def check(self):
        return self.verify(self.path)

    def read(self):
        self._data = self.read_function(self.path)
        return self.data

    def write(self):
        if self.type == 'null':
            return
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))
        self.write_function(self._data, self.path)


class Task:
    def __init__(self, info: dict, assigned_gpu=None):
        task_name = info['name']
        self.name = task_name
        self.type = info['type']
        self.generate_function = task_definition[self.type]['function']
        self.input_target = {k: Target(t) for k, t in info['input_targets'].items()}
        self.output_target = [Target(t) for t in info['output_targets']]
        self.parameters = info['parameters']
        self.resource = task_definition[self.type]['resource']
        self.assigned_gpu = assigned_gpu

    def check_output(self):
        if len(self.output_target) == 0:
            return False
        for t in self.output_target:
            if not t.check():
                return False
        return True

    def check_input(self):
        for t in self.input_target.values():
            if not t.check():
                return False
        return True

    def execute(self):
        if self.assigned_gpu is not None:
            torch.cuda.set_device(self.assigned_gpu)
        input_list = {k: t.read() for k, t in self.input_target.items()}
        result = self.generate_function(**input_list, **self.parameters)
        if isinstance(result, tuple):
            for i in range(len(result)):
                self.output_target[i].set_data(result[i])
        elif result is None:
            pass
        else:
            self.output_target[0].set_data(result)
        for t in self.output_target:
            t.write()


def execute_process(t, assigned_gpu=None):
    task = Task(t, assigned_gpu)
    task.execute()
    del task
    gc.collect()


class Executor:
    def __init__(self, pipe=None):
        self.use_multiprocess = True
        self.resource_amount = resource_amount
        self.resource_occupation = [0 for i in resource_amount]
        self.gpu_occupation = {i: None for i in range(resource_amount[1])}
        self.path = None
        self.start_time = None
        self.tasks = {}
        self.targets = {}
        self.unfinished_tasks = set()
        self.unfinished_targets = set()
        self.running_tasks = {}
        self.metadata = {}
        self.raw_data_info = {}
        self.input_info = {}

        self._pool = None
        self.pipe = pipe

    def set_tasks(self, input_info):
        self.input_info = input_info
        self.path = input_info['path']
        self.metadata = input_info['metadata']
        if 'raw_data_info' in input_info:
            self.raw_data_info = input_info['raw_data_info']
        task_list = input_info['tasks']
        for i in task_list:
            t = Task(task_list[i])
            self.tasks[i] = t
            self.unfinished_tasks.add(i)
            for k in t.input_target:
                self.targets[k] = t.input_target[k]
            for r in t.output_target:
                if r.name not in self.targets:
                    self.targets[r.name] = r.name
                self.unfinished_targets.add(r.name)

    def send_message(self, message):
        if self.pipe is not None:
            self.pipe.send({'message': message})
        else:
            print(message)

    def _update_metadata(self, task_name):
        t_ = self.tasks[task_name]
        for r in t_.output_target:
            if r.name == 'null':
                continue
            self.unfinished_targets.remove(r.name)
            d = self.metadata
            for i in range(len(r.category)):
                k = r.category[i]
                if k not in d:
                    d[k] = {}
                d = d[k]
            k = r.path
            if self.path is not None and r.path is not None:
                k = os.path.relpath(r.path, os.path.join(self.path, self.input_info['name'], r.category[0]))
            d[k] = r.metadata

    def _start_task(self, task_name):
        self.send_message('[{}] Start {}'.format(time.asctime(), task_name))
        assigned_gpu = None
        if self.tasks[task_name].resource[1] > 0:
            for i in self.gpu_occupation:
                if self.gpu_occupation[i] is None:
                    assigned_gpu = i
                    self.gpu_occupation[i] = task_name
        p = multiprocessing.Process(target=execute_process, args=[self.input_info['tasks'][task_name], assigned_gpu])
        p.start()
        self.running_tasks[task_name] = p
        #self._pool.apply_async(execute_process, args=[self.input_info['tasks'][task_name], assigned_gpu])
        for r in range(len(self.resource_occupation)):
            self.resource_occupation[r] += self.tasks[task_name].resource[r]

    def _task_finished(self, task_name):
        self.send_message('[{}] Finish {}'.format(time.asctime(), task_name))
        t_ = self.tasks[task_name]
        self.running_tasks.pop(task_name)
        for r in range(len(self.resource_occupation)):
            self.resource_occupation[r] -= t_.resource[r]
        for i in self.gpu_occupation:
            if self.gpu_occupation[i] == task_name:
                self.gpu_occupation[i] = None
        self._update_metadata(task_name)
        self.save_metadata(None, os.path.join(self.path, '{}.visor'.format(self.input_info['name'])))
        if self.pipe is not None:
            self.pipe.send({'progress': 1 - len(self.unfinished_targets) / len(self.targets)})

    def _task_skipped(self, task_name):
        self.send_message('Task {} finnished, skip'.format(task_name))
        t = self.tasks[task_name]
        self.unfinished_tasks.remove(task_name)
        self._update_metadata(task_name)
        if self.pipe is not None:
            self.pipe.send({'progress': 1 - len(self.unfinished_targets) / len(self.targets)})

    def _task_failed(self, task_name):
        self.send_message('[{}] Task {} failed'.format(time.asctime(), task_name))

    def _execute_finished(self):
        klist = [k for k in self.running_tasks.keys()]
        for k in klist:
            try:
                self.running_tasks[k].join()
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback)
                self._task_failed(k)
            self._task_finished(k)
        t1 = time.time() - self.start_time
        self.save_metadata(None, os.path.join(self.path, '{}.visor'.format(self.input_info['name'])))
        self.send_message('Total time: {} seconds'.format(t1))

    def execute(self):
        #self._pool = multiprocessing.Pool(processes=resource_amount[0], maxtasksperchild=1)
        #from .executer_fix import NoDaemonProcess
        #self._pool = multiprocessing.Pool(processes=resource_amount[0], maxtasksperchild=1)
        #self._pool.apply_async() = NoDaemonProcess
        self._pool = []
        if self.pipe is not None:
            self.pipe.send({'status': 'Running'})

        self.start_time = time.time()
        for k in self.tasks:
            t = self.tasks[k]
            if t.check_output():
                self._task_skipped(t.name)
                continue
            else:
                while 1:
                    pop_list = []
                    for k in self.running_tasks:
                        self.running_tasks[k].join(0.001)
                        ec = self.running_tasks[k].exitcode
                        if ec is None:
                            continue
                        elif ec != 0:
                            #exc_type, exc_value, exc_traceback = sys.exc_info()
                            #traceback.print_exception(exc_type, exc_value, exc_traceback)
                            self._task_failed(k)
                            self._execute_finished()
                            if self.pipe is not None:
                                self.pipe.send({'status': 'Failed'})
                            raise Exception()
                        pop_list.append(k)
                    for k in pop_list:
                        self._task_finished(k)
                    try:
                        if not t.check_input():
                            raise AssertionError
                        for n in t.input_target.values():
                            if n.name in self.unfinished_targets:
                                raise AssertionError
                        for r in range(len(self.resource_occupation)):
                            if t.resource[r] + self.resource_occupation[r] > self.resource_amount[r] and self.resource_occupation[r] > 0:
                                raise AssertionError
                        break
                    except AssertionError:
                        pass
                    if self.pipe is not None:
                        if self.pipe.poll(1):
                            s = self.pipe.recv()
                            if 'stop' in s:
                                self.pipe.send({'status': 'Stopping'})
                                self._execute_finished()
                                self.pipe.send({'status': 'Stopped'})
                                return
                    else:
                        time.sleep(1)
            self._start_task(t.name)
        self._execute_finished()
        if self.pipe is not None:
            self.pipe.send({'status': 'Finished'})

    def save_metadata(self, input_path, save_path):
        name = self.input_info['name']
        if input_path is None:
            visor_info = {}
        else:
            with open(input_path) as f:
                visor_info = json.load(f)
        visor_info[name] = {}

        path = os.path.join(self.path, name, 'ReconstructionInput.json')
        try:
            assert os.path.getmtime(path) > self.start_time
        except :
            with open(path, 'w') as f:
                json.dump(self.input_info, f, indent=2)
        visor_info[name]['ReconstructionInput'] = os.path.relpath(path, os.path.dirname(save_path))

        path = os.path.join(self.path, name, 'Parameters.json')
        try:
            assert os.path.getmtime(path) > self.start_time
        except :
            with open(path, 'w') as f:
                json.dump(self.input_info['parameters'], f, indent=2)
        visor_info[name]['Parameters'] = os.path.relpath(path, os.path.dirname(save_path))

        for k in self.metadata:
            path = os.path.join(self.path, name, k)
            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(self.path, name, k, '{}.json'.format(k))
            with open(path, 'w') as f:
                json.dump(self.metadata[k], f, indent=2)
            visor_info[name][k] = os.path.relpath(path, os.path.dirname(save_path))
        visor_info = {**visor_info, **self.raw_data_info}
        with open(save_path, 'w') as f:
            json.dump(visor_info, f, indent=2)


def main(input_file, pipe=None):
    if multiprocessing.get_start_method() != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    input_info = json.loads(input_file)
    executor = Executor(pipe)
    executor.set_tasks(input_info)
    try:
        executor.execute()
    except Exception as e:
        print(e)
        raise e


if __name__ == '__main__':
    file = open(sys.argv[1])
    file = file.read()
    main(file)
