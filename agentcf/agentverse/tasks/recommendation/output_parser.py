from __future__ import annotations

import re
import json
from typing import Union

from agentverse.parser import OutputParser, LLMResult

# from langchain.schema import AgentAction, AgentFinish
from agentverse.utils import AgentAction, AgentFinish

from agentverse.parser import OutputParserError, output_parser_registry


@output_parser_registry.register("recommender")
class RecommenderParser(OutputParser):
    def parse(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        try:
            ans_begin = cleaned_output.index('Choice') + len('Choice:')
        except:
            print(cleaned_output)
            print("!!!!!")
        ans_end = cleaned_output.index('Explanation')
        rat_begin = cleaned_output.index('Explanation') + len('Explanation:')
        ans = cleaned_output[ans_begin:ans_end].strip()
        rat = cleaned_output[rat_begin:].strip()
        if ans == '' or rat == '':
            raise OutputParserError(text)
        return ans, rat

    def parse_backward(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        # ans_begin = cleaned_output.index('Reasons: ') + len('Reasons: ')
        # ans_end = cleaned_output.index('Reflections')
        rat_begin = cleaned_output.index('Updated Strategy') + len('Updated Strategy:')
        # ans = cleaned_output[ans_begin:ans_end].strip()
        rat = cleaned_output[rat_begin:].strip()
        return rat


    def parse_summary(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        return cleaned_output.strip()

    def parse_evaluation(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        # print(cleaned_output)
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        # return cleaned_output

        try:
            ans_begin = cleaned_output.index('Rank:') + len('Rank:')
        except:
            print(cleaned_output)
        # ans_end = cleaned_output.index('Rationale')
        # rat_begin = cleaned_output.index('Rationale') + len('Rationale: ')
        ans = cleaned_output[ans_begin:].strip().split('\n')
        return ans



@output_parser_registry.register("useragent")
class UserAgentParser(OutputParser):
    def parse(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        return cleaned_output.strip()


    def parse_summary(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        return cleaned_output.strip()

    def parse_update(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        try:
            rat_begin = cleaned_output.index('My updated self-introduction') + len('My updated self-introduction:')
        except:
            print(cleaned_output)
        rat = cleaned_output[rat_begin:].strip()
        return rat


@output_parser_registry.register("itemagent")
class ItemAgentParser(OutputParser):
    def parse(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        
        # 尝试标准格式解析
        try:
            ans_begin = cleaned_output.index('The updated description of the first CD') + len(
                'The updated description of the first CD') + 4
            ans_end = cleaned_output.index('The updated description of the second CD')
            rat_begin = cleaned_output.index('The updated description of the second CD') + len(
                'The updated description of the second CD') + 4
            ans = cleaned_output[ans_begin:ans_end].strip()
            rat = cleaned_output[rat_begin:].strip()
            if ans != '' and rat != '':
                return ans, rat
        except Exception as e:
            print(f"Standard parsing failed: {e}")
            print(f"Response content: {cleaned_output}")
        
        # 尝试中文格式解析（智谱GLM可能使用中文输出）
        try:
            if '第一张CD的更新描述' in cleaned_output and '第二张CD的更新描述' in cleaned_output:
                ans_begin = cleaned_output.index('第一张CD的更新描述') + len('第一张CD的更新描述') + 1
                ans_end = cleaned_output.index('第二张CD的更新描述')
                rat_begin = cleaned_output.index('第二张CD的更新描述') + len('第二张CD的更新描述') + 1
                ans = cleaned_output[ans_begin:ans_end].strip()
                rat = cleaned_output[rat_begin:].strip()
                if ans != '' and rat != '':
                    return ans, rat
        except Exception as e:
            print(f"Chinese parsing failed: {e}")
        
        # 尝试简化格式解析（寻找任何描述性内容）
        try:
            lines = cleaned_output.split('\n')
            descriptions = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('The updated') and not line.startswith('第') and len(line) > 10:
                    descriptions.append(line)
            
            if len(descriptions) >= 2:
                return descriptions[0], descriptions[1]
            elif len(descriptions) == 1:
                return descriptions[0], descriptions[0]  # 使用相同描述
        except Exception as e:
            print(f"Simplified parsing failed: {e}")
        
        # 最后的回退方案
        print(f"All parsing methods failed for: {cleaned_output}")
        default_desc = f"Default item description (parsing failed): {cleaned_output[:100]}..."
        return default_desc, default_desc

    def parse_pretrain(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        ans_begin = cleaned_output.index('CD Description: ') + len('CD Description: ')
        ans = cleaned_output[ans_begin:].strip()
        return ans

    def parse_aug(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        ans_begin = cleaned_output.index('Speculated CD Reviews: ') + len('Speculated CD Reviews: ')
        ans = cleaned_output[ans_begin:].strip()
        return ans


