#!/usr/bin/env python3

from typing import Optional, Any


import cv2

import rospy

import numpy as np
import os
from sensor_msgs.msg import CompressedImage, Image
from duckietown.dtros import DTROS, DTParam, NodeType, TopicType
from dt_class_utils import DTReminder
from turbojpeg import TurboJPEG
from cv_bridge import CvBridge





class TestNode(DTROS):
    def __init__(self, node_name):
        super().__init__(node_name, node_type=NodeType.PERCEPTION)

        print(os.getcwd())

        # parameters
        self.publish_freq = DTParam("~publish_freq", -1)

        # utility objects
        self.bridge = CvBridge()
        self.reminder = DTReminder(frequency=self.publish_freq.value)

        # subscribers
        self.sub_img = rospy.Subscriber(
            "~image_in", CompressedImage, self.cb_image, queue_size=1, buff_size="10MB"
        )

        # publishers
        self.pub_img = rospy.Publisher(
            "~image_out",
            Image,
            queue_size=1,
            dt_topic_type=TopicType.PERCEPTION,
            dt_healthy_freq=self.publish_freq.value,
            dt_help="Raw image",
        )

    def cb_image(self, msg):
        # make sure this matters to somebody

        img = self.bridge.compressed_imgmsg_to_cv2(msg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # turn 'raw image' into 'raw image message'
        out_msg = self.bridge.cv2_to_imgmsg(img, "rgb8")
        # maintain original header
        out_msg.header = msg.header
        # publish image
        self.pub_img.publish(out_msg)

if __name__ == "__main__":
    node = TestNode("test_node")
    rospy.spin()