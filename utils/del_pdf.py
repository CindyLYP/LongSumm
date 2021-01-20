import re
import os
import sys
from PyPDF2 import PdfFileReader

#参数为pdf文件全路径名


class DelBrokenPDF:
    def __init__(self, path):
        self.filter=[".pdf"] #设置过滤后的文件类型 当然可以设置多个类型
        self.root_path = path

    def isValidPDF_pathfile2(self, pathfile):
        bValid = True
        try:
            # PdfFileReader(open(pathfile, 'rb'))
            reader = PdfFileReader(pathfile)
            if reader.getNumPages() < 1:  # 进一步通过页数判断。
                bValid = False
        except:
            bValid = False
            # print('*' + traceback.format_exc())

        return bValid

    def isValidPDF_pathfile(self, pathfile):
        r"""
        直接用文件内容判断头尾，
        参数为pdf文件全路径名
        """
        content = ''
        with open(pathfile, mode='rb') as f:
            content = f.read()
        partBegin = content[0:20]
        if partBegin.find(rb'%PDF-1.') < 0:
            print('Error: not find %PDF-1.')
            return False

        idx = content.rfind(rb'%%EOF')
        if idx < 0:
            print('Error: not find %%EOF')
            return False

        partEnd = content[(0 if idx - 100 < 0 else idx - 100): idx + 5]
        if not re.search(rb'startxref\s+\d+\s+%%EOF$', partEnd):
            print('Error: not find startxref')
            return False

        return True

    def all_path(self, dirname):

        result = []#所有的文件

        for maindir, subdir, file_name_list in os.walk(dirname):

            # print("1:",maindir) #当前主目录
            # print("2:",subdir) #当前主目录下的所有目录
            # print("3:",file_name_list)  #当前主目录下的所有文件

            for filename in file_name_list:
                apath = os.path.join(maindir, filename)#合并成一个完整路径
                ext = os.path.splitext(apath)[1]  # 获取文件后缀 [0]获取的是除了文件名以外的内容

                if ext in self.filter:
                    result.append(apath)

        return result

    def start(self):
        # root_path = "../abstractive_papers"

        print(self.root_path)
        dir_or_files = self.all_path(self.root_path)
        with open(os.path.join(self.root_path, "result.txt"), "w") as fp:
            for dir_file in dir_or_files:
                # 获取目录或者文件的路径
                # dir_file_path = os.path.join(root_path,dir_file)
                # print(dir_file)
                if self.isValidPDF_pathfile(dir_file):
                    if not self.isValidPDF_pathfile2(dir_file):
                        print("ERROR2: ", dir_file)
                        fp.write("ERROR\t" + dir_file + "\r\n")
                        os.remove(dir_file)
                else:
                    print("ERROR: ", dir_file)
                    fp.write("ERROR\t" + dir_file + "\r\n")
                    os.remove(dir_file)


