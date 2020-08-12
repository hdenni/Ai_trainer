import json
import cv2
import numpy as np

class draw:
    #################################################################
    #               This class is for drawing skeleton              #
    #################################################################

    ############################## Input ############################
    # fname: json file name                                         #
    # json file ex. {"MidHip_x":0, "MidHip_y":0, "RHip":0, ...}     #
    ############################## Return ###########################
    # dictionary data set(for draw skeleton image)                  #
    #################################################################
    def open_file(fname):
        path = "resource/"
        with open(path + fname, "r") as file:
            temp = json.load(file)

        data = dict()
        for i, item in enumerate(temp.items()):
            if i % 2 == 0:
                s = item[0][:-2]
                data[s] = (int(item[1]),)
            else:
                data[s] += (int(item[1]),)

        del_list = ['LEye', 'REye', 'REar', 'LEar']
        for d in del_list:
            del data[d]

        return data

    ############################## Input ############################
    # data: dictionary data which return by open_file func          #
    # if t==1: highlight leg angle                                  #
    # elif t==2: hightlight waist angle                             #
    # else: X hightlight                                            #
    ############################## Return ###########################
    # Skeleton Image                                                #
    #################################################################
    def draw_skeleton(data, t):
        lines = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
                 [10, 11], [11, 20], [20, 18], [20, 19], [8, 12], [12, 13], [13, 14], [17, 16], [17, 15]]
        l_one = [[9, 10], [10, 11], [12, 13], [13, 14]]
        l_two = [[1, 8], [8, 9], [8, 12]]

        dot_one = [9, 10, 11, 12, 13, 14]
        dot_two = [1, 8, 9, 12]

        blue = (255, 0, 0)
        green = (0, 255, 0)
        red = (0, 0, 255)
        white = (255, 255, 255)
        black = (0, 0, 0)
        yellow = (0, 255, 255)

        # Canvas(white)
        img = np.zeros((1000, 2000, 3), np.uint8) + 255

        # draw line
        data_enum = list(data.values())
        for l in lines:
            cv2.line(img, data_enum[l[0]], data_enum[l[1]], black, 4, lineType=cv2.LINE_4)

        # draw circle
        for d in data.values():
            cv2.circle(img, d, 8, yellow, thickness=-1, lineType=cv2.FILLED)

        if t == 1:
            for l in l_one:
                cv2.line(img, data_enum[l[0]], data_enum[l[1]], red, 5, lineType=cv2.LINE_AA)
            for d in dot_one:
                cv2.circle(img, data_enum[d], 8, red, thickness=-1, lineType=cv2.FILLED)
        elif t == 2:
            for l in l_two:
                cv2.line(img, data_enum[l[0]], data_enum[l[1]], red, 5, lineType=cv2.LINE_AA)
            for d in dot_two:
                cv2.circle(img, data_enum[d], 8, red, thickness=-1, lineType=cv2.FILLED)

        return img

    # Trim Image function
    ############################## Input ############################
    # data: dictionary data which return by open_file func          #
    # img: image data whith return by drawk_skeleton func           #
    ############################## Return ###########################
    # trimed skeleton image                                         #
    #################################################################
    def img_trim(data, img):
        w = 300; h = 500;
        x = data['MidHip'][0]
        y = data['MidHip'][1]

        return img[y-h:y+h, x-w:x+w]

	# save img file at resource folder
	# file will be jpg file(you can change filename extension)
	def make_img(img, filename):
		cv2.imwrite('resource/'+filename+'.jpg', img)

