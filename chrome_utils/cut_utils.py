def scaleCuts(img):
    height, width, channels = img.shape
    ##### We ensure the cuts are of a decent size
    if height * width < 32 * 32: #We remove very small cuts 
        return None
    return img