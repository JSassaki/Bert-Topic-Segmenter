import check_diff
import topic_segmenter
import os

def check_all_differences():
    for archive in os.listdir("fulltexts/"):
        check_diff.segmentation_difference(archive)


def segment_all_texts():
    for archive in os.listdir("fulltexts/"):
        topic_segmenter.segment_topics(archive)


def label_all_segmented():
    for archive in os.listdir("fulltexts/"):
        topic_segmenter.create_label(archive)


if __name__ == '__main__':
    segment_all_texts()
    check_all_differences()
    label_all_segmented()
