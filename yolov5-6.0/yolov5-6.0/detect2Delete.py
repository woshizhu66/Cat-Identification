import os
import shutil


def delete(origin, future):
    future = os.listdir(future)

    flist = []
    for file in future:
        flist.append(str(file).strip('.jpg'))

    origin = os.listdir(origin)

    for i in origin:
        i = str(i).strip('.xml')
        if i not in flist:
            # os.remove(os.path.join(origin, '/', i))
            print(i)
        else:
            continue


delete(r'D:\project\cat_sum\Annotation', r'D:\System\yolov5-6.0\yolov5-6.0\data\images')
