import cv2
import numpy as np

def get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):

    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def main(img_path):

	img = cv2.imread(img_path)
	input_res = 384

	height, width = img.shape[0], img.shape[1]
	c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
	s = max(img.shape[0], img.shape[1]) * 1.0
	rot = 10

	s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
	w_border = get_border(128, img.shape[1])
	h_border = get_border(128, img.shape[0])
	c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
	c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

	trans_input = get_affine_transform(
	  c, s, rot, [input_res, input_res])

	inp = cv2.warpAffine(img, trans_input, 
	  (input_res, input_res),
	   flags=cv2.INTER_LINEAR)

	cv2.imshow("Imagen Transformada",inp)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__=='__main__':
	img_path = "./FLIR_01698.jpeg"
	main(img_path)