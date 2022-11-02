import check_diff
import topic_segmenter
import os
import topic_labeler
import sys

def check_all_differences(folder):
    precision, recall, f1, windiff, pk,i =0,0,0,0,0,0
    smallest,biggest,mean=99,0,0
    c=sys.argv[2]
    with open("performance/performance_9-11",'w+') as file:
        for archive in getListOfFiles(folder):
            if archive[-3:]!="ted" and "9-11" in archive:
                i+=1
                pr, r, f, w, p, b, s, m=check_diff.segmentation_difference(archive,c)
                precision+=pr
                recall+=r
                f1+=f
                windiff+=w
                pk+=p
                if smallest > s:
                    smallest=s
                if biggest < b:
                    biggest=b
                mean+=m
                file.write(archive + "\nprecision: " + '%.2f' % (pr*100) +"\nrecall:: "+ '%.2f' % (r*100) +"\nf1: "+ '%.2f' % (f*100) +"\nwindiff: "+ '%.2f' % (w * 100)+'%' +"\npk:"+ '%.2f' % (p * 100)+'%'+"\n\n")
        file.write("Total\nprecision: "+ "%.3f" % (precision*100/i)+ "%\nrecall: "+ "%.3f" % (recall*100/i)+"%\nf1-score "+ "%.3f" % (f1*100/i)+"%\nwindiff: "+"%.3f" % (windiff/i)+"\npk: "+"%.3f" % (pk/i)+"\nbiggest: "+str(biggest)+"\nsmallest: "+str(smallest)+"\nmean: "+str(mean/i))
    print("precision:", "%.3f" % (precision*100/i), "%\trecall:", "%.3f" % (recall*100/i), "%\tf1-score", "%.3f" % (f1*100/i), "%\nwindiff:", "%.3f" % (windiff/i), "\tpk:","%.3f" % (pk/i),"\nbiggest:", biggest, "\tsmallest:", smallest, "\tmean:",mean/i)
    # check_diff.segmentation_difference("brown")


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def segment_all_texts(folder):
    c=sys.argv[2]
    time=0
    for archive in getListOfFiles(folder):
        print("segmenting",archive)
        time+=topic_segmenter.segment_topics(archive,c)
    # topic_segmenter.segment_topics("brown")
    print("Total execution time:",time/700)

def label_all_segmented(folder):
    for archive in os.listdir(folder):
        print("labeling", folder + archive)
        topic_labeler.create_label(archive)


if __name__ == '__main__':
    folder=sys.argv[1]
    # segment_all_texts(folder)
    #check_all_differences(folder)
    label_all_segmented(folder)
    # print(sys.argv[1])
