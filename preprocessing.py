from copy import deepcopy
from typing import Union, Tuple, List
import numpy as np
import torch
from einops import rearrange
from torch.nn import functional as F
from abc import ABC, abstractmethod
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
from os.path import join, isdir, isfile, dirname
import os


# THE UTILS FUNCTION COME FROM THE OFFICIAL GITHUB OF THE NNUNET IMPLEMENTATION


ANISO_THRESHOLD = 3


class BaseReaderWriter(ABC):
    @staticmethod
    def _check_all_same(input_list):
        # compare all entries to the first
        for i in input_list[1:]:
            if i != input_list[0]:
                return False
        return True

    @staticmethod
    def _check_all_same_array(input_list):
        # compare all entries to the first
        for i in input_list[1:]:
            if i.shape != input_list[0].shape or not np.allclose(i, input_list[0]):
                return False
        return True

    @abstractmethod
    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        """
        Reads a sequence of images and returns a 4d (!) np.ndarray along with a dictionary. The 4d array must have the
        modalities (or color channels, or however you would like to call them) in its first axis, followed by the
        spatial dimensions (so shape must be c,x,y,z where c is the number of modalities (can be 1)).
        Use the dictionary to store necessary meta information that is lost when converting to numpy arrays, for
        example the Spacing, Orientation and Direction of the image. This dictionary will be handed over to write_seg
        for exporting the predicted segmentations, so make sure you have everything you need in there!

        IMPORTANT: dict MUST have a 'spacing' key with a tuple/list of length 3 with the voxel spacing of the np.ndarray.
        Example: my_dict = {'spacing': (3, 0.5, 0.5), ...}. This is needed for planning and
        preprocessing. The ordering of the numbers must correspond to the axis ordering in the returned numpy array. So
        if the array has shape c,x,y,z and the spacing is (a,b,c) then a must be the spacing of x, b the spacing of y
        and c the spacing of z.

        In the case of 2D images, the returned array should have shape (c, 1, x, y) and the spacing should be
        (999, sp_x, sp_y). Make sure 999 is larger than sp_x and sp_y! Example: shape=(3, 1, 224, 224),
        spacing=(999, 1, 1)

        For images that don't have a spacing, set the spacing to 1 (2d exception with 999 for the first axis still applies!)

        :param image_fnames:
        :return:
            1) a np.ndarray of shape (c, x, y, z) where c is the number of image channels (can be 1) and x, y, z are
            the spatial dimensions (set x=1 for 2D! Example: (3, 1, 224, 224) for RGB image).
            2) a dictionary with metadata. This can be anything. BUT it HAS to include a {'spacing': (a, b, c)} where a
            is the spacing of x, b of y and c of z! If an image doesn't have spacing, just set this to 1. For 2D, set
            a=999 (largest spacing value! Make it larger than b and c)

        """
        pass

    @abstractmethod
    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        """
        Same requirements as BaseReaderWriter.read_image. Returned segmentations must have shape 1,x,y,z. Multiple
        segmentations are not (yet?) allowed

        If images and segmentations can be read the same way you can just `return self.read_image((image_fname,))`
        :param seg_fname:
        :return:
            1) a np.ndarray of shape (1, x, y, z) where x, y, z are
            the spatial dimensions (set x=1 for 2D! Example: (1, 1, 224, 224) for 2D segmentation).
            2) a dictionary with metadata. This can be anything. BUT it HAS to include a {'spacing': (a, b, c)} where a
            is the spacing of x, b of y and c of z! If an image doesn't have spacing, just set this to 1. For 2D, set
            a=999 (largest spacing value! Make it larger than b and c)
        """
        pass

    @abstractmethod
    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        """
        Export the predicted segmentation to the desired file format. The given seg array will have the same shape and
        orientation as the corresponding image data, so you don't need to do any resampling or whatever. Just save :-)

        properties is the same dictionary you created during read_images/read_seg so you can use the information here
        to restore metadata

        IMPORTANT: Segmentations are always 3D! If your input images were 2d then the segmentation will have shape
        1,x,y. You need to catch that and export accordingly (for 2d images you need to convert the 3d segmentation
        to 2d via seg = seg[0])!

        :param seg: A segmentation (np.ndarray, integer) of shape (x, y, z). For 2D segmentations this will be (1, y, z)!
        :param output_fname:
        :param properties: the dictionary that you created in read_images (the ones this segmentation is based on).
        Use this to restore metadata
        :return:
        """
        pass


class SimpleITKIO(BaseReaderWriter):
    supported_file_endings = [
        '.nii.gz',
        '.nrrd',
        '.mha',
        '.gipl'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        spacings = []
        origins = []
        directions = []

        spacings_for_nnunet = []
        for f in image_fnames:
            itk_image = sitk.ReadImage(f)
            spacings.append(itk_image.GetSpacing())
            origins.append(itk_image.GetOrigin())
            directions.append(itk_image.GetDirection())
            npy_image = sitk.GetArrayFromImage(itk_image)
            if npy_image.ndim == 2:
                # 2d
                npy_image = npy_image[None, None]
                max_spacing = max(spacings[-1])
                spacings_for_nnunet.append((max_spacing * 999, *list(spacings[-1])[::-1]))
            elif npy_image.ndim == 3:
                # 3d, as in original nnunet
                npy_image = npy_image[None]
                spacings_for_nnunet.append(list(spacings[-1])[::-1])
            elif npy_image.ndim == 4:
                # 4d, multiple modalities in one file
                spacings_for_nnunet.append(list(spacings[-1])[::-1][1:])
                pass
            else:
                raise RuntimeError(f"Unexpected number of dimensions: {npy_image.ndim} in file {f}")

            images.append(npy_image)
            spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(spacings):
            print('ERROR! Not all input images have the same spacing!')
            print('Spacings:')
            print(spacings)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(origins):
            print('WARNING! Not all input images have the same origin!')
            print('Origins:')
            print(origins)
            print('Image files:')
            print(image_fnames)
            print(
                'It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
                'that segmentations and data overlap.')
        if not self._check_all_same(directions):
            print('WARNING! Not all input images have the same direction!')
            print('Directions:')
            print(directions)
            print('Image files:')
            print(image_fnames)
            print(
                'It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
                'that segmentations and data overlap.')
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing_for_nnunet! (This should not happen and must be a '
                  'bug. Please report!')
            print('spacings_for_nnunet:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        stacked_images = np.vstack(images)
        dict = {
            'sitk_stuff': {
                # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                'spacing': spacings[0],
                'origin': origins[0],
                'direction': directions[0]
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            'spacing': spacings_for_nnunet[0]
        }
        return stacked_images.astype(np.float32), dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname,))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        assert seg.ndim == 3, 'segmentation must be 3d. If you are exporting a 2d segmentation, please provide it as shape 1,x,y'
        output_dimension = len(properties['sitk_stuff']['spacing'])
        assert 1 < output_dimension < 4
        if output_dimension == 2:
            seg = seg[0]

        itk_image = sitk.GetImageFromArray(seg.astype(np.uint8))
        itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
        itk_image.SetOrigin(properties['sitk_stuff']['origin'])
        itk_image.SetDirection(properties['sitk_stuff']['direction'])

        sitk.WriteImage(itk_image, output_fname, True)


def resample_torch_simple(
        data: Union[torch.Tensor, np.ndarray],
        new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
        is_seg: bool = False,
        num_threads: int = 4,
        device: torch.device = torch.device('cpu'),
        memefficient_seg_resampling: bool = False,
        mode='linear'
):
    if mode == 'linear':
        if data.ndim == 4:
            torch_mode = 'trilinear'
        elif data.ndim == 3:
            torch_mode = 'bilinear'
        else:
            raise RuntimeError
    else:
        torch_mode = mode

    if isinstance(new_shape, np.ndarray):
        new_shape = [int(i) for i in new_shape]

    if all([i == j for i, j in zip(new_shape, data.shape[1:])]):
        return data
    else:
        n_threads = torch.get_num_threads()
        torch.set_num_threads(num_threads)
        new_shape = tuple(new_shape)
        with torch.no_grad():

            input_was_numpy = isinstance(data, np.ndarray)
            if input_was_numpy:
                data = torch.from_numpy(data).to(device)
            else:
                orig_device = deepcopy(data.device)
                data = data.to(device)

            if is_seg:
                unique_values = torch.unique(data)
                result_dtype = torch.int8 if max(unique_values) < 127 else torch.int16
                result = torch.zeros((data.shape[0], *new_shape), dtype=result_dtype, device=device)
                if not memefficient_seg_resampling:
                    result_tmp = torch.zeros((len(unique_values), data.shape[0], *new_shape), dtype=torch.float16,
                                             device=device)
                    scale_factor = 1000
                    done_mask = torch.zeros_like(result, dtype=torch.bool, device=device)
                    for i, u in enumerate(unique_values):
                        result_tmp[i] = \
                            F.interpolate((data[None] == u).float() * scale_factor, new_shape, mode=torch_mode, )[0]
                        mask = result_tmp[i] > (0.7 * scale_factor)
                        result[mask] = u.item()
                        done_mask |= mask
                    if not torch.all(done_mask):
                        # print('resolving argmax', torch.sum(~done_mask), "voxels to go")
                        result[~done_mask] = unique_values[result_tmp[:, ~done_mask].argmax(0)].to(result_dtype)
                else:
                    for i, u in enumerate(unique_values):
                        if u == 0:
                            pass
                        result[F.interpolate((data[None] == u).float(), new_shape, mode=torch_mode)[  # antialias
                                   0] > 0.5] = u
            else:
                result = F.interpolate(data[None].float(), new_shape, mode=torch_mode)[0]  # antialias
            if input_was_numpy:
                result = result.cpu().numpy()
            else:
                result = result.to(orig_device)
        torch.set_num_threads(n_threads)
        return result


def get_do_separate_z(spacing: Union[Tuple[float, ...], List[float], np.ndarray], anisotropy_threshold=ANISO_THRESHOLD):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis


def compute_new_shape(old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                      old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                      new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]) -> np.ndarray:
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
    return new_shape


def determine_do_sep_z_and_axis(
        force_separate_z: bool,
        current_spacing,
        new_spacing,
        separate_z_anisotropy_threshold: float = ANISO_THRESHOLD) -> Tuple[bool, Union[int, None]]:
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(current_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            do_separate_z = False
            axis = None
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
            axis = None
        else:
            axis = axis[0]
    return do_separate_z, axis


def resample_torch_fornnunet(
        data: Union[torch.Tensor, np.ndarray],
        new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
        current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
        new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
        is_seg: bool = False,
        num_threads: int = 4,
        device: torch.device = torch.device('cpu'),
        memefficient_seg_resampling: bool = False,
        force_separate_z: Union[bool, None] = None,
        separate_z_anisotropy_threshold: float = ANISO_THRESHOLD,
        mode='linear',
        aniso_axis_mode='nearest'
):
    """
    data must be c, x, y, z
    """
    assert data.ndim == 4, "data must be c, x, y, z"
    new_shape = [int(i) for i in new_shape]
    orig_shape = data.shape

    do_separate_z, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                                      separate_z_anisotropy_threshold)

    if do_separate_z:
        was_numpy = isinstance(data, np.ndarray)
        if was_numpy:
            data = torch.from_numpy(data)

        axis = [axis]
        assert len(axis) == 1
        axis = axis[0]
        tmp = "xyz"
        axis_letter = tmp[axis]
        others_int = [i for i in range(3) if i != axis]
        others = [tmp[i] for i in others_int]

        # reshape by overloading c channel
        data = rearrange(data, f"c x y z -> (c {axis_letter}) {others[0]} {others[1]}")

        # reshape in-plane
        tmp_new_shape = [new_shape[i] for i in others_int]
        data = resample_torch_simple(data, tmp_new_shape, is_seg=is_seg, num_threads=num_threads, device=device,
                                     memefficient_seg_resampling=memefficient_seg_resampling, mode=mode)
        data = rearrange(data, f"(c {axis_letter}) {others[0]} {others[1]} -> c x y z",
                         **{
                             axis_letter: orig_shape[axis + 1],
                             others[0]: tmp_new_shape[0],
                             others[1]: tmp_new_shape[1]
                         }
                         )
        # reshape out of plane w/ nearest
        data = resample_torch_simple(data, new_shape, is_seg=is_seg, num_threads=num_threads, device=device,
                                     memefficient_seg_resampling=memefficient_seg_resampling, mode=aniso_axis_mode)
        if was_numpy:
            data = data.numpy()
        return data
    else:
        return resample_torch_simple(data, new_shape, is_seg, num_threads, device, memefficient_seg_resampling)




#os.mkdir("/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new")




path_to_imgs = "/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed"
df = pd.read_csv(join(path_to_imgs, '1_subjects_label_hcc_deeptekcalv_chic.csv'),
                 delimiter=",").drop(['Unnamed: 0'], axis=1).reset_index(drop=True)

df = df[df['is_registered'] == 1]
subjects = list(list(df['datapoint'].unique()) +
                [x.replace('.nii.gz','_ART.nii.gz') for x in list(
                    df[df['base'] == 'deeptek']['datapoint'].unique())])
u = []

for x in tqdm(subjects):
    if x.startswith('bat'):
        if x.replace('.nii.gz', '_DEL.nii.gz') in os.listdir(
                "/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/6_deeptek_ct_hcc__volume"):
            u.append(x.replace('.nii.gz', '_DEL.nii.gz'))
        if x.replace('.nii.gz', '_PRE.nii.gz') in os.listdir(
                "/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/6_deeptek_ct_hcc__volume"):
            u.append(x.replace('.nii.gz', '_PRE.nii.gz'))

    else:
        if x.replace('VEN.nii.gz', 'DEL.nii.gz') in os.listdir(
                "/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/3_calv_ct_hcc__volume"):
            u.append(x.replace('VEN.nii.gz', 'DEL.nii.gz'))
        if x.replace('VEN.nii.gz', 'PRE.nii.gz') in os.listdir(
                "/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/3_calv_ct_hcc__volume"):
            u.append(x.replace('VEN.nii.gz', 'PRE.nii.gz'))


subjects = subjects+u
# change to the new target spacing
target_spacing = [2., 0.76171899, 0.76171899]       # target spacing: median of all the image spacings

io = SimpleITKIO()


print(len(subjects))
print(len(np.unique(np.array(subjects))))


for sub in tqdm(np.unique(np.array(subjects))):

    #if isdir(join(
    #    "/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new",
    #    sub.replace('.nii.gz','.npy'))):
    #    print('next iter')
    #    continue

    if sub.startswith('bat'):
        path_to_imgs = "/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/6_deeptek_ct_hcc__volume"
    else:
        path_to_imgs = "/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/3_calv_ct_hcc__volume"

    img_file, pkl = io.read_images((join(path_to_imgs, sub), ))
    spacing = pkl['spacing']
    target_shape = compute_new_shape(np.squeeze(img_file).shape, spacing, target_spacing)

    new_img = resample_torch_fornnunet(img_file, target_shape, spacing, target_spacing, is_seg=False)
    np.save(join(
        "/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new",
        sub.replace('.nii.gz','.npy')), new_img)

    if sub.endswith('_ART.nii.gz') and sub.startswith('bat'):   # if deeptek: same segmentation for art and ven so register the seg only once
        continue
    if sub.endswith('_DEL.nii.gz') and sub.startswith('bat'):   # if deeptek: same segmentation for art and ven so register the seg only once
        continue
    if sub.endswith('_PRE.nii.gz') and sub.startswith('bat'):   # if deeptek: same segmentation for art and ven so register the seg only once
        continue

    #### REPLACE LESION BY LIVER AND LAUNCH THE CODE TO HAVE THE PREPROCESSED LIVERSEGS IN JEANZAY

    seg_file, _ = io.read_images((join(path_to_imgs, sub.replace('.nii.gz','__liver.nii.gz')),))
    new_seg = resample_torch_fornnunet(seg_file, target_shape, spacing, target_spacing, is_seg=True)
    np.save(join(
        "/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new",
        sub.replace('.nii.gz','__liver.npy')), new_seg)