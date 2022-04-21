import os

for filename in os.listdir('D:/project/cat_sum/Annotation/'):
    newname = filename.replace(' ', '_')  # 把logo-替换成logo-abc-
    os.rename('D:/project/cat_sum/Annotation/' + filename, 'D:/project/cat_sum/Annotation/' + newname)
