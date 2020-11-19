#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, SegmentList, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading

from lane_controller.controller import PurePursuitLaneController


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocitie.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        # Add the node parameters to the parameters dictionary
        self.params = dict()
        self.follow_point = [0,0]
        self.look_ahead_distance = 0.0
        self.sin_alpha = 0
        self.v = 0.1
        self.omega = 0
        self.d = 0
        self.phi = 0
        self.pp_controller = PurePursuitLaneController(self.params)

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber("~lane_pose",
                                                 LanePose,
                                                 self.cbLanePoses,
                                                 queue_size=1)

        self.sub_segment_list = rospy.Subscriber("/agent/lane_filter_node/seglist_filtered",
                                                 SegmentList,
                                                 self.cbControllerProcessSegments,
                                                 queue_size=1)                                         

        self.log("Lane Controller Initialized!")
        

    def cbLanePoses(self, input_pose_msg):
        """Callback receiving pose messages

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        # print("IN LANE POSES CALLBACK")
        self.pose_msg = input_pose_msg
        self.d = self.pose_msg.d 
        self.phi = self.pose_msg.phi 

        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.pose_msg.header

        # TODO This needs to get changed
        self.publishCmd(car_control_msg)

    def cbControllerProcessSegments(self, input_segment_list):
        """Callback receiving pose messages

        Args:
            input_segment_list (:obj:`LineDectecorNode`): Message containing information about the segment list.
        """
        print("IN PROCESS SEGMENTS CALLBACK")

        white_segments_x, white_segments_y, yellow_segments_x, yellow_segments_y = self.extractSegmentsOfInterest(input_segment_list.segments)    
        white_segments_length = len(white_segments_x)
        yellow_segments_length = len(yellow_segments_x)

        if (len(white_segments_x)>=1 and len(yellow_segments_x)>=1 and (len(white_segments_x)+len(yellow_segments_x))>=5):

            #Compute avg white segment coordinates 
            x_avg_white = np.sum(np.array(white_segments_x))/white_segments_length
            y_avg_white = np.sum(np.array(white_segments_y))/white_segments_length
        
            #Compute avg yellow segment coordinates
            x_avg_yellow = np.sum(np.array(yellow_segments_x))/yellow_segments_length
            y_avg_yellow = np.sum(np.array(yellow_segments_y))/yellow_segments_length
        
            #Compute follow point
            follow_point_x = (x_avg_white+x_avg_yellow)/2.0
            follow_point_y = (y_avg_white+y_avg_yellow)/2.0
        
            #Update parameters
            self.follow_point = [follow_point_x, follow_point_y]
            self.look_ahead_distance = np.sqrt(follow_point_x**2 + follow_point_y**2)
            self.sin_alpha = self.follow_point[1] / self.look_ahead_distance
            self.omega = (2.0 * self.v * self.sin_alpha) / self.look_ahead_distance
            self.v = 0.5
        else:
            print("USE LANE POSE 2")
            print("PHI2=", self.phi)
            print("DISTANCE FROM MIDDLE OF LANE2", self.d)
            self.omega = -self.phi
            self.v = 0.1


    def extractSegmentsOfInterest(self, segments):
        #Extract white segments coordinates
        white_segments_x = [seg.points[1].x for seg in segments if (seg.color==0 and seg.points[1].y < 0)]
        white_segments_y = [seg.points[1].y for seg in segments if (seg.color==0 and seg.points[1].y < 0)]    
        #Extract yellow segments coordinates
        yellow_segments_x = [seg.points[1].x for seg in segments if (seg.color==1 and seg.points[1].y > 0)]
        yellow_segments_y = [seg.points[1].y for seg in segments if (seg.color==1 and seg.points[1].y > 0)]       
        return [white_segments_x, white_segments_y, yellow_segments_x, yellow_segments_y]
    
    
    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        print("OMEGA=", self.omega)
        print("V=", self.v)
        print("----------------------------------------------------------------------------------------------")
        car_cmd_msg.omega = self.omega
        car_cmd_msg.v = self.v
        # car_cmd_msg.omega = 0.5
        # car_cmd_msg.v = 0.1     
        self.pub_car_cmd.publish(car_cmd_msg)


    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()