from follow_up.InterviewQuestionAnalyzer import InterviewQuestionAnalyzer
from follow_up.FollowUpQuestionGenerator import FollowUpQuestionGenerator
from collections import defaultdict


class FlowManager:

    def __init__(self, analyzer: InterviewQuestionAnalyzer, generator: FollowUpQuestionGenerator):
        self.previous_question = []
        self.history = defaultdict(list)  # (초기 질문 a, 초기 답변 a,분석 a) = [(꼬리 질문 b, 꼬리 답변 b, 분석 b), ...]
        self.analyzer = analyzer
        self.generator = generator

    def start_flow(self):

        while True:
            question_input = input("질문을 입력 하세요.").strip()  # 패딩 제거 안되면 이상 해짐.
            answer_input = input("답변을 입력 하세요.").strip()

            if question_input == "n" or answer_input == "n":
                print('end!')
                break
            else:

                self.previous_question.append(question_input)

                analysis = self.analyzer.start_analysis(question=question_input, answer=answer_input)

                print("<<first follow up>>", end='')
                follow_up_question = self.generator.generate_follow_up(question=question_input, answer=answer_input,
                                                                       analysis=analysis,
                                                                       previous_question=self.previous_question)
                self.previous_question.append(follow_up_question)

                follow_up_answer = input("답변을 입력 하세요.").strip()

                follow_up_analysis = self.analyzer.start_analysis(question=follow_up_question, answer=follow_up_answer)

                self.history[(question_input, answer_input, analysis)].append(
                    (follow_up_question, follow_up_answer, follow_up_analysis))

                print("<<second follow up>>", end='')
                another_follow_up_question = self.generator.generate_follow_up(question=follow_up_question,
                                                                               answer=follow_up_answer,
                                                                               analysis=follow_up_analysis,
                                                                               previous_question=self.previous_question)

                self.previous_question.append(another_follow_up_question)

                another_follow_up_answer = input("답변을 입력 하세요.").strip()
                another_follow_up_analysis = self.analyzer.start_analysis(question=another_follow_up_question,
                                                                          answer=another_follow_up_answer)
                self.history[(question_input, answer_input, analysis)].append(
                    (another_follow_up_question, another_follow_up_answer, another_follow_up_analysis))

        print()
        print(self.previous_question)
        print()
        for key in self.history.keys():
            print(key)
            for value in self.history[key]:
                print(value)
