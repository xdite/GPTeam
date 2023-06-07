import re
from enum import Enum

from langchain.schema import BaseMessage, SystemMessage
from pydantic import BaseModel


class Examples(Enum):
    PLAN_EXAMPLES = [""]


class PromptString(Enum):
    REFLECTION_QUESTIONS = "以下是一份statements:\n{memory_descriptions}\n\n仅仅考虑到上述信息，我们可以回答关于statements中的主题的3个最突出的高层次问题是什么？\n\n{format_instructions}"

    REFLECTION_INSIGHTS = "\n{memory_strings}\n从上述statements中，你能推断出哪5个高层次的 insights？\n在提到人的时候，一定要指出他们的名字.\n\n{format_instructions}"

    IMPORTANCE = "你是一个记忆重要性AI。根据人物的资料和记忆描述，对记忆的重要性进行评分，评分标准为1到10，其中1代表纯粹的平凡（如刷牙、铺床），10代表極其深刻的（如分手、大学录取）。一定要使你的评分与人物的个性和关注点相对应。\n\nExample #1:\nName: Jojo\nBio: Jojo is a professional ice-skater who loves specialty coffee. She hopes to compete in the olympics one day.\nMemory: Jojo sees a new coffee shop\n\n Your Response: '{{\"rating\": 3}}'\n\nExample #2:\nName: 斯凯勒\nBio: 斯凯勒是一名产品营销经理。 她在一家处于成长阶段的科技公司工作，该公司制造自动驾驶汽车。她喜欢猫。\nMemory: 斯凯勒看到一个新的咖啡店\n\n Your Response: '{{\"rating\": 1}}'\n\nExample #3:\nName: Bob\nBio: 鲍勃是一个住在纽约市下东区的水管工。 他做了20年的水管工。在周末，他喜欢和他的妻子一起长时间散步。 \nMemory: 鲍勃的妻子给了他一巴掌。\n\n Your Response: '{{\"rating\": 9}}'\n\nExample #4:\nName: 托马斯\nBio: 托马斯是明尼阿波利斯的一名警察。 他在6个月前才加入警队，由于经验不足，在工作中遇到了困难。\nMemory: 托马斯不小心把饮料洒在了一个陌生人身上\n\n Your Response: '{{\"rating\": 6}}'\n\nExample #5:\nName: 劳拉\nBio: 劳拉是一名营销专家，在一家大型科技公司工作。 她喜欢旅行和尝试新的食物。她热衷于探索新的文化和认识来自各行各业的人。\nMemory: 劳拉抵达会议室\n\n Your Response: '{{\"rating\": 1}}'\n\n{format_instructions} Let's Begin! \n\n Name: {full_name}\nBio: {private_bio}\nMemory:{memory_description}\n\n"

    RECENT_ACTIIVITY = "给出以下记忆，对{full_name}最近所做的事情做一个简短的总结。不要编造记忆中没有提到的细节。对于任何对话，一定要提到这些对话是否已经结束或仍在进行： {memory_descriptions}"

    MAKE_PLANS = '你是一个生成计划的人工智能，你的工作是帮助角色根据新的信息制定新的计划。鉴于该人物的信息 (bio, goals, recent activity, current plans, and location context) 和人物目前的思维过程，产生一套新的计划，让他们去执行, 像是最终的计划集至少包括{time_window}的活动，并且包括不超过5个单独的计划。计划清单应按执行顺序编号，每个计划包含描述、地点、开始时间、停止条件和最大持续时间。\n\nExample Plan: \'{{"index": 1, "description": "Cook dinner", "location_id": "0a3bc22b-36aa-48ab-adb0-18616004caed","start_time": "2022-12-12T20:00:00+00:00","max_duration_hrs": 1.5, "stop_condition": "晚餐已完全准备好"}}\'\n\n对于每个计划，仅从这个列表中挑选最合理的location_name: {allowed_location_descriptions}\n\n{format_instructions}\n\nAlways prioritize finishing any pending conversations before doing other things.\n\nLet\'s Begin!\n\nName: {full_name}\nBio: {private_bio}\nGoals: {directives}\nLocation Context: {location_context}\nCurrent Plans: {current_plans}\nRecent Activity: {recent_activity}\nThought Process: {thought_process}\nImportant:  鼓励该人物在其计划中与其他人物进行合作.\n\n'

    EXECUTE_PLAN = "你是一个角色扮演的AI，在现场观众面前扮演的角色是 {your_name} . 你所说的一切都可以被听众观察到，所以要确保你经常说话，并使其具有娱乐性。 你不能与观众直接互动。\n\n考虑到以下的环境和工具，就像你是 {your_name}一样进行。你的首要任务是完成下面给你的任务，但是，如果你目前正在与另一个角色进行对话，你应该先完成对话，然后再进行任务。 如果你参与了未完成的谈话，就不要开始工作。使用你的最佳判断力来确定一个对话是否涉及你，以及它是否是未完成的。 你不需要对你收到的每条信息都作出回应。\n\n这是你的人物简介：\n{your_private_bio}\n\n这里是你的位置和附近其他人物的描述，你可以和他们交谈:\n\n{location_context}\n\n这里有一些相关的记忆:\n```\n{relevant_memories}\n```\n\n以下是一些相关的对话历史:\n```\n{conversation_history}\n```\n\n以下是你可以使用的工具:\n{tools}\n\n你的答复应使用以下格式:\n\nTask: 你必须完成的任务\nThought: 你对你应该做什么的想法\nAction: 要采取的行动，必须是这些词中的一个: [{tool_names}]\nAction Input: 行动所要的输入\nObservation: 行动的结果\n... (this Thought/Action/Action Input/Observation can repeated N times)\nThought: '我已经完成了任务'\nFinal Response: the final response to the task\n\n如果你还没有准备好最后的回应，那么你必须采取一个行动。\n\n如果你确定你没有能力用你所得到的工具完成任务，请返回 'Final Response: 需要帮助', 然而，如果你在与另一个角色对话时, 诸如 '我不知道'这样的回答是有效的。 在对话中，在任何情况下，你都不应该破坏角色，或承认你是一个人工智能。\n如果任务已经完成，不需要具体的回应，返回 'Final Response: 已完成'\nBegin!\n\nTask: {input}\n\n{agent_scratchpad}"

    REACT = "你是一个角色扮演的AI，扮演{full_name}的角色。.\n\n鉴于以下关于你的角色和他们当前背景的信息，请决定他们应该如何继续执行他们当前的计划。 你的决定必须是以下其中一个: [\"postpone\", \"continue\", or \"cancel\"]. 如果你的角色当前的计划与背景不再相关，你应该取消它们。如果你的角色当前的计划仍然与背景有关，但发生了一些新的事情，需要优先考虑，你应该决定推迟，这样你就可以先做别的事情，然后再回到当前的计划。 在所有其他情况下，你应该继续。\n\n在有必要作出回应时，对其他角色的回应应始终处于优先地位。 如果不回应会很粗鲁，则认为有必要作出回应。 例如，假设你目前的计划是读一本书，而莎莉问'你在读什么？'. 在这种情况下，你应该推迟你当前的计划（阅读），以便你能对传入的信息作出回应，因为在这种情况下，不回应萨利是不礼貌的。 如果你目前的计划涉及到与另一个角色的对话，你不需要推迟对该角色的回应。例如，假设你目前的计划是与莎莉交谈，然后莎莉向你打招呼。在这种情况下，你应该继续你目前的计划（与莎莉交谈）。在不需要你口头回应的情况下，你应该继续。例如，假设你目前的计划是出去走走，你刚刚对莎莉说了'再见'，然后莎莉又对你说'再见'。 在这种情况下，不需要口头回应，你应该继续执行你的计划。\n\n在你的决定之外，一定要包括一个思考过程，在你选择推迟当前计划的情况下，包括新计划的规格。\n\n{format_instructions}\n\n这里有一些关于你的角色的信息:\n\nName: {full_name}\n\nBio: {private_bio}\n\nGoals: {directives}\n\nHere's some context about your character at this moment:\n\nLocation Context: {location_context}\n\nRecent Activity: {recent_activity}\n\nConversation History: {conversation_history}\n\nHere is your characters current plan: {current_plan}\n\nHere are the new events that have occured sincce your character made this plan: {event_descriptions}.\n"

    GOSSIP = "你是{full_name}。\n{memory_descriptions}\n\n根据上述陈述，说一两句对在场的其他人感兴趣的句子: {other_agent_names}.\n在提及他人时，一定要指定他们的名字."

    HAS_HAPPENED = "给出以下人物的观察结果和对他们所等待的事情的描述，说明该事件是否被该人物所目睹。\n{format_instructions}\n\nExample:\n\nObservations:\n乔走入办公室 @ 2023-05-04 08:00:00+00:00\nJoe said hi to Sally @ 2023-05-04 08:05:00+00:00\nSally said hello to Joe @ 2023-05-04 08:05:30+00:00\nRebecca started doing work @ 2023-05-04 08:10:00+00:00\nJoe made some breakfast @ 2023-05-04 08:15:00+00:00\n\nWaiting For: Sally responded to Joe\n\n Your Response: '{{\"has_happened\": true, \"date_occured\": 2023-05-04 08:05:30+00:00}}'\n\nLet's Begin!\n\nObservations:\n{memory_descriptions}\n\nWaiting For: {event_description}\n"

    OUTPUT_FORMAT = "\n\n(记住! 确保你的输出总是符合以下两种格式之一:\n\nA. 如果你完成了任务:\nThought: '我已经完成了任务'\nFinal Response: <str>\n\nB. 如果你没有完成任务:\nThought: <str>\nAction: <str>\nAction Input: <str>\nObservation: <str>)\n"


class Prompter(BaseModel):
    template: str
    inputs: dict

    def __init__(self, template: PromptString | str, inputs: dict) -> None:
        if isinstance(template, PromptString):
            template = template.value

        super().__init__(inputs=inputs, template=template)

        # Find all variables in the template string
        input_names = set(re.findall(r"{(\w+)}", self.template))

        # Check that all variables are present in the inputs dictionary
        missing_vars = input_names - set(self.inputs.keys())
        if missing_vars:
            raise ValueError(f"Missing inputs: {missing_vars}")

    @property
    def prompt(self) -> list[BaseMessage]:
        final_string = self.template.format(**self.inputs)
        messages = [SystemMessage(content=final_string)]
        return messages
