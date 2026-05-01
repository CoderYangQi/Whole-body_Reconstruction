import os, subprocess

ILASTIK_EXECUTABLE = '"C:/Program Files/ilastik-1.3.2rc2/ilastik.exe"'


def run_ilastik(project_file, image_file, output_file):
    command = '{} --headless --project={} ' \
              '--export_source="simple segmentation" ' \
              '--output_format="multipage tiff" ' \
              '--output_filename_format={} {}'\
        .format(ILASTIK_EXECUTABLE, project_file, output_file, image_file)
    print(command)
    subprocess.call(command)

