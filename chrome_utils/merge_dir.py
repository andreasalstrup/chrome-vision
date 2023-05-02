import os
import shutil

class MergeDir:
    def __init__(self, src_dir, dst_dir, dst_index):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.dst_index = dst_index
        if not (os.path.exists(dst_dir) | os.path.exists(dst_index)):
            self.__merge_dir(self.src_dir, self.dst_dir, self.dst_index)

    def __merge_dir(self, src_dir, dst_dir, dst_index):
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        for item in os.listdir(src_dir):
            if os.path.isdir(os.path.join(src_dir, item)):
                self.__merge_dir(f"{src_dir}/{item}", dst_dir, dst_index)
            else:
                file = f"{src_dir}/{item}"
                shutil.copy(file, dst_dir)
                self.__createIndexFile(item, dst_index)

    def __createIndexFile(self, item, path):
        if not os.path.exists(f"{path}"):
            file = open(f"{path}", "x")
        
        with open(f"{path}", "a") as file:
            file.write(item + '\n')

        file.close()