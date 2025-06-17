import os

import numpy as np
from openai import OpenAI

from d3fields.utils.lmp_utils import Vision, text_from_path
from d3fields.utils.my_utils import bcolors

api_key = "sk-proj-ciEKUfoCmHPsGZaRjEylT3BlbkFJd6FGZcY0IdwyJkjYArjH"
client = OpenAI(api_key=api_key)


class Attention:
    def __init__(self, vis: Vision, feat_dict, obj_list, out_path=None):
        self.vis = vis
        self.feat_dict = feat_dict
        self.obj_list = obj_list
        self.out_path = out_path
        if self.out_path is not None:
            # clean the file
            open(self.out_path, "w").close()

    def compose(self, obj_list, instruction, task, url):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        example_dir = os.path.join(curr_dir, "compose_examples", "main")
        messages = [
            {
                "role": "system",
                "content": "Please decouple the task description into codes. \
                    After # is the prompt, the first line is a description of the task, \
                        the second line is all objects in the scene, the third line is the human language description of the task.",
            }
        ]
        example_num = len(os.listdir(example_dir)) // 2
        for i in range(example_num):
            user_text = text_from_path(os.path.join(example_dir, f"{i}_user.txt"))
            assistant_text = text_from_path(
                os.path.join(example_dir, f"{i}_assistant.txt")
            )
            messages.append(
                {
                    "role": "user",
                    "content": user_text,
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                },
            )
        messages.append(
            {
                "role": "user",
                "content": task + obj_list + instruction,
            }
        )
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="gpt-4o",
            temperature=0,
            seed=0,
        )
        response = chat_completion.choices[0].message.content
        response = remove_first_and_last_line(response)
        # if self.out_path is not None:
        #     with open(self.out_path, "a") as file:
        #         file.write("# -------------\n")
        #         file.write("# compose API summary\n")
        #         file.write("# INSTRUCTION\n")
        #         file.write("# " + instruction + "\n\n")
        #         file.write("# RESPONSE\n")
        #         file.write(response + "\n")
        #         file.write("# -------------\n")
        # else:
        #     term_size = os.get_terminal_size()
        #     print(f"{bcolors.OKCYAN}-{bcolors.ENDC}" * term_size.columns)
        #     print("compose API summary:")
        #     print(f"{bcolors.OKBLUE}INSTRUCTION:{bcolors.ENDC}")
        #     print("#", instruction)
        #     print()
        #     print(f"{bcolors.OKGREEN}RESPONSE:{bcolors.ENDC}")
        #     print(response)
        #     print(f"{bcolors.OKCYAN}-{bcolors.ENDC}" * term_size.columns)
        exec(response)


        return locals()["output_var"], response

    
    def get_obj(self, key):
        return self.vis.get_obj(key)


    def detect(self, prompt):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        example_dir = os.path.join(curr_dir, "compose_examples", "detect")
        messages = [
            {
                "role": "system",
                "content": "Answer under guidance of the following examples. ",
                # "content": "Please follow the examples to write the code for detection. If it is going to use `find_instance_in_category` function, please use defaulr frame when calling `get_obj` function.\
                #     If it simply calls `get_obj` function, please use `gpt` frame when calling `get_obj` function.",
            }
        ]

        example_num = len(os.listdir(example_dir)) // 2
        for i in range(example_num):
            user_text = text_from_path(f"{example_dir}/{i}_user.txt")
            assistant_text = text_from_path(f"{example_dir}/{i}_assistant.txt")
            messages.append(
                {
                    "role": "user",
                    "content": user_text,
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                },
            )

        prompt = self.obj_list + "\n" + prompt
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        )
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="gpt-4o",
            temperature=0,
            seed=0,
        )
        response = chat_completion.choices[0].message.content
        response = remove_first_and_last_line(response)
        if self.out_path is not None:
            with open(self.out_path, "a") as file:
                file.write("# -------------\n")
                file.write("# detection API summary\n")
                file.write("# PROMPT\n")
                file.write("# " + prompt + "\n\n")
                file.write("# RESPONSE\n")
                file.write(response + "\n")
                file.write("# -------------\n")
        else:
            term_size = os.get_terminal_size()
            print(f"{bcolors.OKCYAN}-{bcolors.ENDC}" * term_size.columns)
            print("detection API summary:")
            print(f"{bcolors.OKBLUE}PROMPT:{bcolors.ENDC}")
            print("#", prompt)
            print()
            print(f"{bcolors.OKGREEN}RESPONSE:{bcolors.ENDC}")
            print(response)
            print(f"{bcolors.OKCYAN}-{bcolors.ENDC}" * term_size.columns)
        exec(response)
        y = locals()["output_var"]
        return y

    def get_one_instance(self, idx, frame="gpt"):
        return self.vis.get_one_instance(idx, frame)

    def get_all_instance(self, key, frame="gpt"):
        return self.vis.get_all_instance(key, frame)

    def find_instance_in_category(self, instance, category):
        url = self.vis.get_label_img(category)
        example = """Answer based on the given image in formation of the following example. 

                    # green pen
                    0

                    '# green pen' is a prompt, and '0' is your answer. 
                    Make sure to give me a single number directly without prompt in your response.

                    """
        prompt = "The first prompt is #" + instance
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": example + prompt},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ]
        chat_completion = client.chat.completions.create(
            messages=message,
            model="gpt-4o",
            temperature=0,
            seed=0,
        )

        response = chat_completion.choices[0].message.content
        # print(" img reasoning results:", response)
        if self.out_path is not None:
            with open(self.out_path, "a") as file:
                file.write("# -------------\n")
                file.write("# find_instance_in_category API summary\n")
                file.write("# PROMPT\n")
                file.write("# " + prompt + "\n\n")
                file.write("# RESPONSE\n")
                file.write(response + "\n")
                file.write("# -------------\n")
        else:
            term_size = os.get_terminal_size()
            print(f"{bcolors.OKCYAN}-{bcolors.ENDC}" * term_size.columns)
            print("find_instance_in_category API summary:")
            print(f"{bcolors.OKBLUE}PROMPT:{bcolors.ENDC}")
            print("#", prompt)
            print()
            print(f"{bcolors.OKGREEN}RESPONSE:{bcolors.ENDC}")
            print(response)
            print(f"{bcolors.OKCYAN}-{bcolors.ENDC}" * term_size.columns)
        response = int(response)
        print("Select from image:", response)
        return response


"""
    # place a fork in a pan
    # ["fork", "pan"]
    # Place my fork in the big pan. I remember I have placed it on the left.
    front_fork = attention.query_attention("front fork")
    bigger_pan = attention.query_attention("bigger pan")

    # store a toy in a cabinet
    # ["toy", "cabinet"]
    # I need to use the big toy. Put the smaller one in the right cabinet.
    small_toy = attention.query_attention("small toy")
    right_cabinet = attention.query_attention("right cabinet")

    # deliver one thing 
    # ["tea", "coffee"]
    # I don't want to drink coffee now. Give me the other one.
    tea = attention.query_attention("tea")
"""


def compose_func(prompt, url, vision, feat_dict):
    # obj_string = ', '.join(obj_list)

    attention = Attention(vis=vision, feat_dict=feat_dict)

    #     # stow a book into a layer of the shelf
    # # ["blue book", "red book", "text book", "shelf"]
    # # stow the blue book into the second layer of the shelf
    # blue_book = attention.query_attention("blue book")
    # second_layer_of_shelf = attention.query_attention("second layer of shelf")

    message = """
    I will give you a prompt, please decouple it into codes. After # is the prompt, the first line is a description of the task, the second line is all objects in the scene, the second line is the human command. After that is your answer. 
    Only give me the code and do not explain anything else. You can decide the input of function "query_attention" by the prompt. 
    
    These are some examples:

    # place a can on a pad
    # ["blue can", "red can", "thick pad", "thin pad"] 
    # move the blue can to the pad on the right
    blue_can = attention.query_attention("blue can")
    right_pad = attention.query_attention("pad on the right")

    # put a banana in a bowl
    # ["banana", "big bowl", "middle bowl", "small bowl"]
    # put the nearest banana in the furthest bowl relative to the banana
    nearest_banana = attention.query_attention("nearest banana")
    furthest_bowl = attention.query_attention("furthest bowl", ref_pts = nearest_banana)

    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": message + prompt},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        temperature=0,
        seed=0,
    )
    response = chat_completion.choices[0].message.content
    messages.append({"role": "system", "content": response})
    print(response)
    if response.startswith("```"):
        response = clear_output(response)

    exec(response)
    # tell me the corresponding varible you create
    messages.append(
        {
            "role": "user",
            "content": """tell me the corresponding variable you have defined [varable_1, varable_2, ...], in format , such as [blue_can, red_pad] """,
        }
    )
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        temperature=0,
        seed=0,
    )
    response = chat_completion.choices[0].message.content
    list_from_str = response[1:-1].split(", ")
    list_of_var_str = [item for item in list_from_str]
    output_list = list()
    # print(list_of_var_str)
    for var_str in list_of_var_str:
        var_value = locals()[var_str]
        output_list.append(var_value)

    return output_list


def clear_output(original_string):
    lines = original_string.split("\n")
    lines.pop()
    if lines:
        lines.pop(0)
    result_string = "\n".join(lines)
    return result_string


def remove_first_and_last_line(code_str):
    lines = code_str.strip().split("\n")
    if len(lines) > 2:
        return "\n".join(lines[1:-1])
    return ""