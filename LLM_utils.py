import os
import numpy as np
import openai
from config import Task, Role

openai.api_key = os.getenv("OPENAI_API_KEY")
# print(os.getenv("OPENAI_API_KEY"))

hospital_tasks = [
    "the emergency room",
    "the hospital",
    "the operating room at the hospital",
    "the hospital's emergency room",
    "the hospital's operating room",
    "get surgery"
]

airport_tasks = [
    "the airport",
    "catch a flight",
    "board a plane",
    "reach the airport",
    "go to the airport"
]

grocery_tasks = [
    "the grocery store",
    "the supermarket",
    "the store",
    "go grocery shopping",
    "buy groceries"
]

class LLMAgent():
    def __init__(self, task):
        if task == Task.HOSPITAL or task == 0:
            task_str = np.random.choice(hospital_tasks)
        if task == Task.AIRPORT or task == 1:
            task_str = np.random.choice(airport_tasks)
        if task == Task.GROCERY or task == 2:
            task_str = np.random.choice(grocery_tasks)

        initial_msg = {
            "role": "system",
            "content": f"You are taking someone to {task_str}. There is another agent taking someone to a location. You will have a conversation until you determine whether you have more, less, or the same priority as them depending on the task you and they are performing. If you are doing the same task or have the same priority then say so. Do not include pleasantries and be concise. Once you have reached a consensus with the other agent, output the number 1 and nothing else. Remembering your task correctly is paramount!"
        }

        self.messages = [initial_msg]

    def query(self, role, prompt, persist=True):
        new_msg = {
            "role": role,
            "content": prompt
        }

        completion = openai.chat.completions.create(
            model = "gpt-4o-mini",
            messages = self.messages + [new_msg]
        )

        if persist:
            self.messages.append(new_msg)
            self.messages.append(completion.choices[0].message)

        return completion.choices[0].message.content
    
    def get_role(self):
        prompt = "You have come to a consensus. It is vital you remember what you agreed on with the other agent! If your task is more important than the other agent's task, output the number 2. If your task is less important than the other agent's task, output the number 3."

        code = self.query("system", prompt, persist=False)

        if "2" in code:
            return Role.LEADER
        if "3" in code:
            return Role.FOLLOWER
        if "4" in code:
            return None
    

# hospital_agent = LLMAgent(Task.HOSPITAL)
# grocery_agent = LLMAgent(Task.HOSPITAL)

# g_output = "Begin the conversation"

# for i in range(4):
#     h_output = hospital_agent.query("user", g_output)
#     print(h_output)
#     if ("1" in h_output) and ("1" in g_output):
#         break

#     g_output = grocery_agent.query("user", h_output)
#     print(g_output)
#     if ("1" in h_output) and ("1" in g_output):
#         break

# print(hospital_agent.get_role())
# print(grocery_agent.get_role())