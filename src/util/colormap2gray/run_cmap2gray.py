import timeit
import argparse
import colormap2gray_cy

setup = """import colormap2gray_cy
import colormap2gray
import argparse
parser = argparse.ArgumentParser(description='Transform thermal images in colormap to grayscale')
parser.add_argument('--data_path', help='input data path')
parser.add_argument('--img_format',default='png',help='image format')
parser.add_argument('--output_dir',help='output directory for grayscale images')
parser.add_argument('--include_set_name',action='store_true',help="add name of set as preamble of each image")
args = parser.parse_args()
"""
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Transform thermal images in colormap to grayscale')
	parser.add_argument('--data_path', help='input data path')
	parser.add_argument('--img_format',default='png',help='image format')
	parser.add_argument('--output_dir',help='output directory for grayscale images')
	parser.add_argument('--include_set_name',action='store_true',help="add name of set as preamble of each image")
	args = parser.parse_args()
	
	"""
	cy = timeit.Timer('colormap2gray_cy.main(args)',setup=setup)
	#py = timeit.Timer('colormap2gray.main(args)',setup=setup)
	cy_time = cy.timeit(5)
	#py_time = py.timeit(2)
	print("Cython Program Time:", cy_time)
	#print("Python Program Time:", py_time)
	"""
	colormap2gray_cy.main(args)