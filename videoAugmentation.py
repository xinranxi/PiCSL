# ----------------------------------------
# Written by Yuecong Min
# ----------------------------------------
import cv2
import pdb
import PIL
import copy
import scipy.misc
import torch
import random
import numbers
import numpy as np
import torchvision.transforms as transforms


def _resize_numpy_image(img, size, interp='bilinear'):
    pil_img = PIL.Image.fromarray(img)
    resample = {
        'nearest': PIL.Image.NEAREST,
        'lanczos': PIL.Image.LANCZOS,
        'bilinear': PIL.Image.BILINEAR,
        'bicubic': PIL.Image.BICUBIC,
        'cubic': PIL.Image.BICUBIC,
    }.get(interp, PIL.Image.BILINEAR)
    return np.array(pil_img.resize((size[1], size[0]), resample=resample))


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class WERAugment(object):
    def __init__(self, boundary_path):
        self.boundary_dict = np.load(boundary_path, allow_pickle=True).item()
        self.K = 3

    def __call__(self, video, label, file_info):
        ind = np.arange(len(video)).tolist()
        if file_info not in self.boundary_dict.keys():
            return video, label
        binfo = copy.deepcopy(self.boundary_dict[file_info])
        binfo = [0] + binfo + [len(video)]
        k = np.random.randint(min(self.K, len(label) - 1))
        for i in range(k):
            ind, label, binfo = self.one_operation(ind, label, binfo)
        ret_video = [video[i] for i in ind]
        return ret_video, label

    def one_operation(self, *inputs):
        prob = np.random.random()
        if prob < 0.3:
            return self.delete(*inputs)
        elif 0.3 <= prob < 0.7:
            return self.substitute(*inputs)
        else:
            return self.insert(*inputs)

    @staticmethod
    def delete(ind, label, binfo):
        del_wd = np.random.randint(len(label))
        ind = ind[:binfo[del_wd]] + ind[binfo[del_wd + 1]:]
        duration = binfo[del_wd + 1] - binfo[del_wd]
        del label[del_wd]
        binfo = [i for i in binfo[:del_wd]] + [i - duration for i in binfo[del_wd + 1:]]
        return ind, label, binfo

    @staticmethod
    def insert(ind, label, binfo):
        ins_wd = np.random.randint(len(label))
        ins_pos = np.random.choice(binfo)
        ins_lab_pos = binfo.index(ins_pos)

        ind = ind[:ins_pos] + ind[binfo[ins_wd]:binfo[ins_wd + 1]] + ind[ins_pos:]
        duration = binfo[ins_wd + 1] - binfo[ins_wd]
        label = label[:ins_lab_pos] + [label[ins_wd]] + label[ins_lab_pos:]
        binfo = binfo[:ins_lab_pos] + [binfo[ins_lab_pos - 1] + duration] + [i + duration for i in binfo[ins_lab_pos:]]
        return ind, label, binfo

    @staticmethod
    def substitute(ind, label, binfo):
        sub_wd = np.random.randint(len(label))
        tar_wd = np.random.randint(len(label))

        ind = ind[:binfo[tar_wd]] + ind[binfo[sub_wd]:binfo[sub_wd + 1]] + ind[binfo[tar_wd + 1]:]
        label[tar_wd] = label[sub_wd]
        delta_duration = binfo[sub_wd + 1] - binfo[sub_wd] - (binfo[tar_wd + 1] - binfo[tar_wd])
        binfo = binfo[:tar_wd + 1] + [i + delta_duration for i in binfo[tar_wd + 1:]]
        return ind, label, binfo



class ColorJitter(object):
    """
    Applies color jittering to the video frames.
    确保同一个Batch内（即同一段视频序列）采用相同的抖动参数，保持时序一致性。
    """
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.5):
        self.transform = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.p = p

    def __call__(self, video):
        if random.random() > self.p:
            return video

        if isinstance(video, list):
            if len(video) == 0:
                return video
            if isinstance(video[0], np.ndarray):
                video = np.array(video)
            elif isinstance(video[0], PIL.Image.Image):
                # Convert PIL to numpy
                video = np.array([np.array(img) for img in video])

        # If it's a numpy array, shape is supposed to be (T, H, W, C)
        if isinstance(video, np.ndarray):
            # Convert to Tensor (T, C, H, W)
            t = torch.from_numpy(video.transpose((0, 3, 1, 2)))
            # Apply ColorJitter (Torchvision applies same jitter across the batch)
            t = self.transform(t)
            # Convert back to (T, H, W, C) numpy array
            video = t.numpy().transpose((0, 2, 3, 1))
            
        elif isinstance(video, torch.Tensor):
            # If already Tensor, assume (T, C, H, W)
            video = self.transform(video)
            
        return video

class ToTensor(object):
    def __call__(self, video):
        if isinstance(video, list):
            video = np.array(video)
            video = torch.from_numpy(video.transpose((0, 3, 1, 2))).float()
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose((0, 3, 1, 2)))
        return video


class RandomCrop(object):
    """
    Extract random crop of the video.
    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).
        crop_position (str): Selected corner (or center) position from the
        list ['c', 'tl', 'tr', 'bl', 'br']. If it is non, crop position is
        selected randomly at each call.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size

    def __call__(self, clip):
        crop_h, crop_w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if crop_w > im_w:
            pad = crop_w - im_w
            clip = [np.pad(img, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), 'constant', constant_values=0) for img in
                    clip]
            w1 = 0
        else:
            w1 = random.randint(0, im_w - crop_w)

        if crop_h > im_h:
            pad = crop_h - im_h
            clip = [np.pad(img, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)), 'constant', constant_values=0) for img in
                    clip]
            h1 = 0
        else:
            h1 = random.randint(0, im_h - crop_h)

        if isinstance(clip[0], np.ndarray):
            return [img[h1:h1 + crop_h, w1:w1 + crop_w, :] for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.crop((w1, h1, w1 + crop_w, h1 + crop_h)) for img in clip]


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        try:
            im_h, im_w, im_c = clip[0].shape
        except ValueError:
            print(clip[0].shape)
        new_h, new_w = self.size
        new_h = im_h if new_h >= im_h else new_h
        new_w = im_w if new_w >= im_w else new_w
        top = int(round((im_h - new_h) / 2.))
        left = int(round((im_w - new_w) / 2.))
        return [img[top:top + new_h, left:left + new_w] for img in clip]


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, clip):
        # B, H, W, 3
        flag = random.random() < self.prob
        if flag:
            clip = np.flip(clip, axis=2)
            clip = np.ascontiguousarray(copy.deepcopy(clip))
        return np.array(clip)


class RandomRotation(object):
    """
    Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')
        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [np.array(PIL.Image.fromarray(img).rotate(angle)) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return rotated


class TemporalRescale(object):
    def __init__(self, temp_scaling=0.2, frame_interval=1):
        self.min_len = 32 #32
        # self.max_len = int(np.ceil(230/frame_interval))
        self.max_len = int(np.ceil(230 / frame_interval))
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, clip):
        scale = 4
        vid_len = len(clip)
        new_len = int(vid_len * (self.L + (self.U - self.L) * np.random.random()))
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - scale) % scale != 0:
            new_len += scale - (new_len - scale) % scale
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
        return clip[index]


class TemporalDropout(object):
    def __init__(self, drop_ratio=0.1, min_len=32, p=0.5):
        self.drop_ratio = max(0.0, min(0.9, float(drop_ratio)))
        self.min_len = max(1, int(min_len))
        self.p = max(0.0, min(1.0, float(p)))

    def __call__(self, clip):
        if len(clip) <= self.min_len or random.random() > self.p:
            return clip

        vid_len = len(clip)
        keep_len = max(self.min_len, int(round(vid_len * (1.0 - self.drop_ratio))))
        if keep_len >= vid_len:
            return clip

        keep_indices = sorted(random.sample(range(vid_len), keep_len))
        return clip[keep_indices]

class RandomResize(object):
    """
    Resize video bysoomingin and out.
    Args:
        rate (float): Video is scaled uniformly between
        [1 - rate, 1 + rate].
        interp (string): Interpolation to use for re-sizing
        ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic').
    """

    def __init__(self, rate=0.0, interp='bilinear'):
        self.rate = rate
        self.interpolation = interp

    def __call__(self, clip):
        scaling_factor = random.uniform(1 - self.rate, 1 + self.rate)

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_h, new_w)
        if isinstance(clip[0], np.ndarray):
            return [_resize_numpy_image(img, (new_h, new_w), self.interpolation) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.resize(size=(new_w, new_h), resample=self._get_PIL_interp(self.interpolation)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

    def _get_PIL_interp(self, interp):
        if interp == 'nearest':
            return PIL.Image.NEAREST
        elif interp == 'lanczos':
            return PIL.Image.LANCZOS
        elif interp == 'bilinear':
            return PIL.Image.BILINEAR
        elif interp == 'bicubic':
            return PIL.Image.BICUBIC
        elif interp == 'cubic':
            return PIL.Image.CUBIC


class Resize(object):
    """
    Resize video bysoomingin and out.
    Args:
        rate (float): Video is scaled uniformly between
        [1 - rate, 1 + rate].
        interp (string): Interpolation to use for re-sizing
        ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic').
    """

    def __init__(self, rate=0.0, interp='bilinear'):
        self.rate = rate
        self.interpolation = interp

    def __call__(self, clip):
        scaling_factor = self.rate

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        if isinstance(clip[0], np.ndarray):
            return [np.array(PIL.Image.fromarray(img).resize(new_size)) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.resize(size=(new_w, new_h), resample=self._get_PIL_interp(self.interpolation)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

    def _get_PIL_interp(self, interp):
        if interp == 'nearest':
            return PIL.Image.NEAREST
        elif interp == 'lanczos':
            return PIL.Image.LANCZOS
        elif interp == 'bilinear':
            return PIL.Image.BILINEAR
        elif interp == 'bicubic':
            return PIL.Image.BICUBIC
        elif interp == 'cubic':
            return PIL.Image.CUBIC

class Resize1(object):
    """
    Resize video bysoomingin and out.
    Args:
        rate (float): Video is scaled uniformly between
        [1 - rate, 1 + rate].
        interp (string): Interpolation to use for re-sizing
        ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic').
    """

    def __init__(self, size=224, interp='bilinear'):
        self.size = size
        self.interpolation = interp

    def __call__(self, clip):
        new_w = int(self.size)
        new_h = int(self.size)
        new_size = (new_w, new_h)
        if isinstance(clip[0], np.ndarray):
            return [np.array(PIL.Image.fromarray(img).resize(new_size)) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.resize(size=(new_w, new_h), resample=self._get_PIL_interp(self.interpolation)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

    def _get_PIL_interp(self, interp):
        if interp == 'nearest':
            return PIL.Image.NEAREST
        elif interp == 'lanczos':
            return PIL.Image.LANCZOS
        elif interp == 'bilinear':
            return PIL.Image.BILINEAR
        elif interp == 'bicubic':
            return PIL.Image.BICUBIC
        elif interp == 'cubic':
            return PIL.Image.CUBIC

class RandomCat(object):
    def __init__(self, size, p):
        self.size = size
        self.p = p

    def __call__(self, clip):
        try:
            im_h, im_w, im_c = clip[0].shape
        except ValueError:
            print(clip[0].shape)

        hNum = im_h//self.size
        wNum = im_w // self.size

        newClip = []
        for img in clip:
            # cv2.imshow("1", img)
            # cv2.waitKey(0)

            img = img.reshape(hNum, self.size, wNum, self.size, im_c)
            img = img.transpose(0, 2, 1, 3, 4)

            img1 = img[:int(hNum * self.p), :wNum, :, :, :]

            for i in range(wNum):
                newHNum = sorted(random.sample(range(hNum), int(hNum * self.p)))
                img1[:,i,:,:,:] = img[newHNum[:],i,:,:,:]

            img2 = img1[:, :int(wNum * self.p), :, :, :]

            for i in range(int(hNum * self.p)):
                newWNum = sorted(random.sample(range(wNum), int(wNum * self.p)))
                img2[i,:,:,:,:] = img1[i,newWNum[:],:,:,:]

            img2 = img2.transpose(0, 2, 1, 3, 4)
            shape = img2.shape
            newImage = img2.reshape(shape[0] * self.size, shape[2] * self.size, im_c)
            newClip.append(newImage)

            # cv2.imshow("2", newImage)
            # cv2.waitKey(0)

        return newClip


