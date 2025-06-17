from pathlib import Path
from openai import OpenAI
client = OpenAI()


input_text = "I am glad to introduce GLID: Generalizable Imitation Learning with 3D Semantic Fields. \
  In this work, we introduced an imitation learning framework capable of category-level generalization, as shown by the diver set of tasks and objects. \
    For example, we use these three shoes to train the shoe aligning task. \
      Here is an example of the training demo, where we push the shoe to rotate and finally align towards the left. \
    Our framework is trained on three instances, and capable of generalizing to novel instances, including shoes with very different appearances and geometry like high-heels and slippers. \
      We incorporate spatial and semantic information using 3D semantic fields. \
        Here is an example of shoe aligning task. \
          In one policy rollout step, we first obtain multi-view RGBD observations and extract 3D descriptor fields encoding high-dimensional features. \
            Compared with selected descriptors from reference images, we obtain our semantic fields, measured by the descriptor similarity. \
              Semantic fields and raw point cloud are then fed into our PointNet++ and diffusion policy network to predict actions. \
                Here is the visualization of our pipeline. Given the observation, we extract the semantic fields as highlighted on the knife handle. \
                  Then the policy predicts actions given current 3D semantic fields and point cloud. Finally, the robot executes predicted actions. \
                  After one rollout step is finished, we repeat the process of obtaining observation, extracting 3D semantic features, predicting and executing actions till the task is finished. \
                    As shown in this knife collecting example, our framework iteratively predicts actions and executes them. Finally it moves towards the container and put the knife down. This shows our whole pipeline. \
                    Next, we show the category-level generalization capability of our framework. \
                      We train our policy on the short white spoon and long silver spoon. \
                       This demo shows how the robot uses spoons to scoop the coffee beans out of the bowl and transport them into the mug. This is challenging since it involves different materials such as granular objects. \
                        Then we evaluate our method on unseen spoons as shown in the grid here. We could see that spoons differ not only in the appearance but also in the shape and size. For example, the measuring spoon in the bottom left is much wider than other spoons."
speech_file_path = Path(__file__).parent / "glid_1.mp3"

response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input=input_text,
)

response.stream_to_file(speech_file_path)

input_text = "Our framework can also resolve geometric ambiguities. \
    For two knives with opposite directions, our framework can localize the knife handles using 3D semantic fields and predict different actions accordingly. \
      This is very challenging to distinguish just from geometric information. \
        Without semantic information, the robot might predict dangerous actions, such as grasping the blade. \
          Our framework pays attention to the subtle geometric details. \
            To place the lying soda can on the table upright, it is necessary to know which side is the top. \
              Certain details could help the robot to make decisions, such as the soda can tab, which, however, is not obvious from geometric information. \
                Our 3D semantic fields attend to such details so that it could place the soda can upright without flipping it. \
                  Another example is spreading the toothpaste on the toothbrush. The robot need to know where is the head to properly spread the toothpaste. \
                    We could see that our policy's predicted actions are diverged to two directions respectively, depending on the toothbrush's head position. \
                      We also note that our policy successfully put toothbrush close to the toothpaste, and finally place the toothbrush on top of the mug. \
                  So, what enables our framework's generalization capability? \
                    Here we show shoes, where shoe heads, shoelace, and shoe tails are highlighted. The highlighting is consistent across different instances and poses. \
                      The consistency across different instances help our framework to generalize. \
                      Our framework also demonstrates its robustness. The shoe aligning and soda can flipping tasks show how the policy could recover from external disturbances and keep making progress with distractor objects. \
                        Thank you for your attention to our framework, GLID. More details about framework is contained in the paper. If you are interested, please refer to it for more details. \
                      "
speech_file_path = Path(__file__).parent / "glid_2.mp3"

response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input=input_text,
)

response.stream_to_file(speech_file_path)