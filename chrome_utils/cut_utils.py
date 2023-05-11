def scaleCuts(img, image_size=32):
    height, width, channels = img.shape
    ##### We ensure the cuts are of a decent size
    if height * width < image_size * image_size: #We remove very small cuts 
        return None
    return img