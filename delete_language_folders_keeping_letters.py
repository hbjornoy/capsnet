import os
import shutil
import exiftool

#copy_tree("/a/b/c", "/x/y/z")

def listdir_nohidden(path):
    l = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            l.append(f)
    return l

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

lang = "language_dataset"s
symb = "symbol_dataset"

for i in listdir_nohidden("language_dataset"):
    for ii in listdir_nohidden('/'.join((lang, i))):
        for iii in listdir_nohidden('/'.join((lang,i,ii))):
            print(ii)
            print(exiftool)
            current_dir = os.getcwd()
            print(current_dir)
            if copytree('/'.join((current_dir, lang, i, ii, iii)), '/'.join((current_dir, lang, i, ii)))