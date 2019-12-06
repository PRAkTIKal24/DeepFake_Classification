# Utility functions for HW7-8
from PIL import Image
from glob import glob
from os.path import basename, exists
from os import mkdir, listdir

mani = 'manipulated'
orig = 'orignal'
def resize_image(path, input_image_path, output_image_path, size=(224, 224)):
    '''Resize image then save.'''
    # read image
    print(input_image_path)
    input_image = Image.open('./' + path+'/'+input_image_path)

    w, h = input_image.size
    y = h / 2 - w / 2
    x = 0

    #cropped image
    cropped_image = input_image.crop((x, y, x+w, y+w))
    # resize image
    element_image = cropped_image.resize((size[0], size[1]), Image.ANTIALIAS)
    # save in output folder
    element_image.save(output_image_path, quality=95)  # 95 is best


def resize_images(input_image_folderpath, path, size=(224, 224)):
    '''Resize all images in folder then save.'''
    # Append folderpath if needed
    if input_image_folderpath.endswith('/') == False:
        print("adding")
        input_image_folderpath = str(input_image_folderpath) + '/'
    output_image_folderpath = './' + path + '-' + 'resize' + '/'
    # Make output folder if it doesn't exist
    if exists(output_image_folderpath) == False:
        print("making")
        mkdir(output_image_folderpath)
    # Build list of input images
    #input_images = glob(input_image_folderpath + '*' + '.jpg')
    input_images = listdir(input_image_folderpath)
    for input_image in input_images:
        out_filename = output_image_folderpath + basename(input_image).split('.')[0] + '.jpg'
        print('Resizing ' + str(out_filename))
        resize_image(path, input_image, out_filename, size=(size[0], size[1]))


if __name__ == '__main__':
    input_image_folderpath = "./manipulated"
    resize_images(input_image_folderpath, mani, size=(224, 224))
    input_image_folderpath = "./orignal"
    resize_images(input_image_folderpath, orig, size=(224, 224))
