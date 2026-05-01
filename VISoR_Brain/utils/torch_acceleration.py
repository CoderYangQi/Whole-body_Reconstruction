import torch
import SimpleITK as sitk
import numpy as np


def resample_affine(image, transform, size, origin, spacing):
    in_origin = image.GetOrigin()
    in_spacing = image.GetSpacing()
    in_size = image.GetSize()
    t = transform.GetParameters()
    transform = np.array([[t[0], t[1], t[2], t[9]],
                          [t[3], t[4], t[5], t[10]],
                          [t[6], t[7], t[8], t[11]],
                          [0, 0, 0, 1]])
    transform = torch.Tensor(transform).cuda()
    image = torch.Tensor(np.float32(sitk.GetArrayFromImage(image))).cuda()

    def block_process(size, origin):
        dims = [torch.full([size[2], size[1], size[0]], spacing[i], device=torch.device('cuda'))
                for i in range(3)]
        length = size[2] * size[1] * size[0]
        out_pos = [torch.reshape(torch.cumsum(dims[i], i), (length,)) + int(origin[2 - i] - spacing[2 - i])
                   for i in range(3)]
        ones = torch.ones((length,), device=torch.device('cuda'))
        out_pos = torch.stack([out_pos[2], out_pos[1], out_pos[0], torch.reshape(ones, (length,))])
        del ones
        in_pos = torch.matmul(transform, out_pos)
        del out_pos
        in_pos = [(in_pos[i] - in_origin[i]) / in_spacing[i] for i in range(3)]
        mask = (torch.ge(in_pos[0], 0) * torch.le(in_pos[0], in_size[0] - 1))
        for i in range(1, 3):
            mask *= (torch.ge(in_pos[i], 0) * torch.le(in_pos[i], in_size[i] - 1))
        mask = mask.float()
        for i in range(3):
            in_pos[i] *= mask
        in_index = in_pos[0].long() + \
                   in_size[0] * in_pos[1].long() + \
                   in_size[0] * in_size[1] * in_pos[2].long()
        del in_pos
        out = torch.take(image, in_index)
        out *= mask
        out = torch.reshape(out, (size[2], size[1], size[0]))
        out = out.cpu().numpy()
        return out

    block_size = [200 * spacing[i] for i in range(3)]
    roi = [origin.copy(), (np.array(size) * np.array(spacing) + np.array(origin)).tolist()]
    out = np.zeros((size[2], size[1], size[0]), np.float32)
    block_count = np.ceil((roi[1][0] - roi[0][0]) / block_size[0]) \
                  * np.ceil((roi[1][1] - roi[0][1]) / block_size[1]) \
                  * np.ceil((roi[1][2] - roi[0][2]) / block_size[2])
    ct = 0

    # Reconstruct image blockwise
    for j in np.arange(roi[0][1], roi[1][1], block_size[1]):
        for i in np.arange(roi[0][0], roi[1][0], block_size[0]):
            for k in np.arange(roi[0][2], roi[1][2], block_size[2]):
                block_roi = [[i, j, k], [i + block_size[0], j + block_size[1], k + block_size[2]]]
                block_roi[1] = np.minimum(block_roi[1], roi[1]).tolist()
                block_image_roi = [np.int32(np.subtract(block_roi[0], roi[0]) / spacing[0]),
                                   np.int32(np.subtract(block_roi[1], roi[0]) / spacing[0])]
                block_image_size = (np.array(block_image_roi[1]) - np.array(block_image_roi[0])).tolist()
                if np.min(block_image_roi[1] - block_image_roi[0]) <= 0:
                    continue
                #print('Generating block {0}/{1}'.format(ct + 1, int(block_count)), *block_image_roi)
                block = block_process(block_image_size, block_roi[0])
                np.copyto(out[block_image_roi[0][2]:block_image_roi[0][2] + block.shape[0],
                              block_image_roi[0][1]:block_image_roi[0][1] + block.shape[1],
                              block_image_roi[0][0]:block_image_roi[0][0] + block.shape[2]], block)
                ct += 1
    out = sitk.GetImageFromArray(out)
    out.SetOrigin(origin)
    out.SetSpacing(spacing)
    return out


if __name__ == '__main__':
    from VISoR_Brain.positioning.visor_sample import VISoRSample
    sl = VISoRSample()
    sl.load('F:/chaoyu/brains/TY_1291/488/2016-10-03_23-08-42.tar')
    sl.load_columns()
    raw = sl.raw_data.load(2, [0, 500])
    t = sl.transforms[0]
    res = resample_affine(raw, t, [1600, 1000, 500], [0, 0, 0], [1, 1, 1])
    sitk.WriteImage(res, 'F:/chaoyu/a.mha')
