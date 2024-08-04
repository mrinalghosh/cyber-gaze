import cv2
import numpy as np
import curses
import time

'''
TODO: 
1. try hex tiling
eg. x x x x x
     x x x x x

2. get the terminal dimensions and draw it with curses

3. colors! 
    just need to send each channel in separately and have some kind of color grading system

4. normalization
    bring down entire intensity by a certain offset to maximize the contrast

5. "move camera" - will need curses
    read arrow keys to pan
    plus minus to zoom
'''


# descending brightness
SCALE_66 = '$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~i!lI;:,"^` '[::-1]
SCALE_29 = '@#W$9876543210?!abc;:+=-,._  '[::-1]
SCALE_10 = "@%#*+=-:. "[::-1]
BYTESIZE = 255
CHANNELS = 3

DEBUG = True


def density_to_char(val: int, scale: str) -> str:
    # suboptimal - use it inline below instead
    return scale[int(val / 255 * (len(scale) - 1))]


def greyscale_to_ascii(img: np.ndarray, scale):
    scale = np.array(list(scale))
    return scale[(img / 255 * (len(scale) - 1)).astype(int)]


def image_to_ascii(img: np.ndarray, width=100, scale=SCALE_10, delimiter=''):
    # takes a greyscale image and converts to an ascii image
    h1, w1 = img.shape
    aspect_ratio = h1 / w1
    height = int(aspect_ratio * width)

    # resize the image
    # always use AREA for shrinking image and LINEAR/CUBIC for enlarging
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    # convert to ascii
    art = greyscale_to_ascii(img, scale)

    # return as a string
    return '\n'.join(delimiter.join(row) for row in art)


def flip(img: np.ndarray):
    return cv2.flip(img, 1)


def color(img: np.ndarray):
    # TODO: average out to greyscale and put in one of the color channels
    # img[:, :, 0] = np.zeros_like(img[:, :, 0]) # B
    img[:, :, 1] = np.zeros_like(img[:, :, 1])  # G
    # img[:, :, 2] = np.zeros_like(img[:, :, 2]) # R

    return img


def average_greyscale(img: np.ndarray):
    # # TODO: buggy
    # alpha = ((img[:, :, 0] + img[:, :, 1] +
    #          img[:, :, 2]) // CHANNELS) % BYTESIZE
    # print(alpha.shape)
    # return alpha

    # from chatgpt
    return img.mean(axis=2)


def cv2_greyscale(img: np.ndarray):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def linear_greyscale_approximation(img: np.ndarray):
    # # TODO: buggy
    # # https://e2eml.school/convert_rgb_to_grayscale
    # approx = 0.299*img[:, :, 2] + 0.587*img[:, :, 1] + 0.114*img[:, :, 0]
    # return np.rint(approx)

    # ... selects all elements with any dimensions - from chatgpt
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


def greyscale(img: np.ndarray):
    # collapse RGB channels to greyscale
    # alpha = average_greyscale(img)
    # alpha = linear_greyscale_approximation(img)
    alpha = cv2_greyscale(img)

    # make n-buckets based on length of string
    # use alpha to index scale
    return alpha


def webcam(width=128, scale=SCALE_29):
    '''
    displays greyscale video and ascii art in standard scrolling terminal output
    '''
    cv2.namedWindow("stream")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        # frams are np.ndarray | height, width, channels - (1080, 1920, 3)
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        frame = flip(frame)
        # cv2.imshow("stream", frame)
        cv2.imshow("stream", greyscale(frame))
        # cv2.imshow("stream", color(frame))

        print(image_to_ascii(greyscale(frame),
              width=width,
              scale=scale,
              delimiter=' '))
        # time.sleep(1)

        rval, frame = vc.read()  # update frame and availability
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    vc.release()
    cv2.destroyWindow("stream")


def cursecam(stdscr):
    # capture webcam input
    video = cv2.VideoCapture(0)

    # grab terminal dimensions
    max_y, max_x = stdscr.getmaxyx()

    stdscr.clear()
    # arbitrary - use the terminal height and width instead
    # height, width = min(max_y, 108//2), min(max_x, 192//2)
    # height, width = 30, 60
    width = max_x//2 - 20

    if video.isOpened():  # try to get the first frame
        # frams are np.ndarray | height, width, channels - (1080, 1920, 3)
        rval, frame = video.read()
    else:
        rval = False

    while rval:
        # flip and recolor
        img = greyscale(flip(frame))

        # generate ascii art
        art = image_to_ascii(img, width, scale=SCALE_66, delimiter=' ')

        # Clear the window and add the ASCII art
        stdscr.clear()
        for i, line in enumerate(art.split('\n')):
            stdscr.addstr(i, 0, f'{i:02} | {line} | {max_x} {max_y}')
            # stdscr.addstr(i, 0, str(i))
        stdscr.refresh()

        rval, frame = video.read()  # update frame and availability

        # exit condition
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break


def get_dimensions(stdscr):
    print(stdscr.getmaxyx())


if __name__ == '__main__':
    # webcam(50, scale=SCALE_10)
    curses.wrapper(cursecam)
    # print('x')
    # curses.wrapper(get_dimensions)
    # print('y')
