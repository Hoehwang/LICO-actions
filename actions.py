# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

import pandas as pd

import numpy as np
import re, random
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# Actions 폴더에 존재하는 Recommend Table을 데이터 프레임의 형태로 불러옴
rec_ta = pd.read_csv('./actions/recommend_table.csv')
# Entity의 분류 양상에 맞춰 컬럼의 이름 및 카테고리 이름을 조정
rec_ta.columns = [re.sub("(design|color|pattern|fabric)\-", "", a) for a in rec_ta.columns.tolist()]
rec_ta["cate"] = [re.sub("WITH\-","",a) for a in rec_ta["cate"].tolist()]

# 인텐트명 및 응답형 출력문의 참조를 위한 리스폰스 테이블을 데이터 프레임 형태로 불러옴
res = pd.read_csv("./actions/RESPONSE_EXP_CLO.csv")

# 모든 토픽어의 종류를 참조하기 위한 테이블을 데이터 프레임 형태로 불러옴
all_features = pd.read_csv("./actions/all_features.csv")
# 토픽어와 토픽어 별 분류 정보를 참조하기 위한 데이터 프레임을 불러옴
syn = pd.read_csv("./actions/SYN.csv")
# 토픽어 이름/노멀라이즈 값 정보 저장
clo_types = list(set(zip(syn[syn["intent"] == "CLOTHES-TYPE"]["entity"].tolist(), syn[syn["intent"] == "CLOTHES-TYPE"]["norm"].tolist())))
# 토픽어 이름만 따로 저장
cl_ty = list(set(syn[syn["intent"] == "CLOTHES-TYPE"]["entity"].tolist()))

class ActionRephraseResponse(Action):
    # 액션에 대한 이름을 설정
    def name(self) -> Text:
        return "action_rephrase_clothes"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # 모델을 통해 분석된 결과를 통해 엔티티 정보를 불러옴
        entity = [[a.get("entity"),a.get("value")] for a in tracker.latest_message["entities"]]
 
        # 모델을 통해 분석된 결과를 통해 인텐트트 정보 불러옴
        inti = tracker.get_intent_of_latest_message()
        print(tracker.get_intent_of_latest_message())

        try:
            clothes_tuple = [a for a in entity if a[0] in cl_ty][0]
            if clothes_tuple[0] == 'UNDERWEAR' and clothes_tuple[1] == '옷':
                clothes = "TSHIRT"
            else:
                clothes = clothes_tuple[0]
        except IndexError:
            clothes = ""
        
        print('clothes: %s' % clothes)
        af4in = re.sub(".+_","",inti)
        entity_wo_clo = [a for a in entity if a in all_features["entity"].tolist()]

        if af4in in ["ALL-FEATURE", "COLOR-TYPE", "DESIGN-TYPE", "PATTERN-TYPE", "FEATURE-GOOD", "FABRIC-TYPE", "FOR-WOMAN"]:
            input_feature = entity_wo_clo

        else:
            input_feature = [re.sub(".+_", "", str(inti))]
        print('input_feature: %s' % input_feature)
        if input_feature == ["NON-FEATURE"] and len(clothes) > 0:

            enti = [b for a, b in clo_types if a == clothes][0]
            enti_n = [a for a, b in clo_types if a == clothes][0]

            ind = [a for a, b in enumerate(res["intent"].tolist()) if str(b) == str(inti)][0]
            results_pop = self.rank_popularity_recommend(enti_n, rec_ta)

            if len(results_pop) > 1:
                dispatcher.utter_message(text="인기있는 " + str(enti) + " 아이템을 보여 드릴게요.")

                num = 1
                for i, t in results_pop:
                    text = str(num) + "위 아이템: " + t
                    num += 1
                    dispatcher.utter_message(text=text)
                    dispatcher.utter_message(image=i)
                dispatcher.utter_message(text=res["utter_ask_more"].tolist()[ind])

            else:
                fallback = "죄송하지만, 원하시는 아이템을 찾을 수 없습니다."
                dispatcher.utter_message(text=fallback)

        elif len(clothes) > 0:

            enti = [b for a, b in clo_types if a == clothes][0]
            enti_n = [a for a, b in clo_types if a == clothes][0]
            ind = [a for a, b in enumerate(res["intent"].tolist()) if str(b) == str(inti)][0]


            fres = '%s' % res["response"].tolist()[ind]
            if len(fres.split(' / ')) > 1:
                fres = random.sample(fres.split(' / '), 1)[0]

            fin = str(fres).replace("<CLOTHES-TYPE_FEATURE>",str(enti))
            for stop_f in ['DESIGN', 'COLOR', 'PATTERN', 'FABRIC']:
                try:
                    input_feature.remove(stop_f)
                except Exception as e:
                    print('Error:', e)
                    continue
            try:
                results = self.rank_recommend(input_feature, enti_n, rec_ta)

                if len(results) > 0:

                    dispatcher.utter_message(text=fin + "\n" + res["utter_send_link"].tolist()[ind])

                    num = 1
                    for i,t in results:
                        text = str(num) + "위 아이템: " + t
                        num += 1
                        dispatcher.utter_message(text=text)
                        dispatcher.utter_message(image=i)
                    dispatcher.utter_message(text=res["utter_ask_more"].tolist()[ind])

                else:

                    results_pop = self.rank_popularity_recommend(enti_n, rec_ta)

                    if len(results_pop) > 1:

                        dispatcher.utter_message(text="죄송하지만, 추천해 드릴 적절한 상품이 없네요.\n대신 인기있는 " + str(enti) + " 아이템을 보여 드릴게요.")

                        num = 1
                        for i, t in results_pop:

                            text = str(num) + "위 아이템: " + t
                            num += 1
                            dispatcher.utter_message(text=text)
                            dispatcher.utter_message(image=i)
                        dispatcher.utter_message(text=res["utter_ask_more"].tolist()[ind])

                    else:
                        fallback = "죄송하지만, 원하시는 아이템을 찾을 수 없습니다."
                        dispatcher.utter_message(text=fallback)
            except:
                fallback = "죄송하지만, 원하시는 아이템을 찾을 수 없습니다."
                dispatcher.utter_message(text=fallback)

        else:
            ind = [a for a, b in enumerate(res["intent"].tolist()) if str(b) == str(inti)][0]
            featureless = res["featureless"].tolist()[ind]

            fin = str(featureless)
            dispatcher.utter_message(text=fin)

        return []

    def rank_recommend(self, input_feature, input_clothes, rec_ta):
        input_feature = [re.sub("(design|color|pattern|fabric)\-", "", a.lower()) for a in input_feature]
        feature = input_feature + ["rank"]
        features = feature + ["sub-category", "cate", "url", "image"]
        rec_ta[feature] = rec_ta[feature].replace(0, np.nan)
        rec_ta = rec_ta[features].dropna()
        rec_ta = rec_ta[features]

        if len(input_feature) > 1:

            for f in feature:
                rec_ta = rec_ta.sort_values(by=f, ascending=False)
                rec_ta[f] = [a for a in range(1, len(rec_ta) + 1)][::-1]

            rec_ta["sum"] = rec_ta[[item for item in feature if item not in feature[-1]]].sum(axis=1)
            rec_ta = rec_ta.sort_values(by="sum", ascending=False)
            print(rec_ta)
        idx = [a for a, b in enumerate(rec_ta["sub-category"].tolist()) if b == input_clothes]
        if len(idx) < 2:
            idx = [a for a, b in enumerate(rec_ta["cate"].tolist()) if b == input_clothes]

        urls = rec_ta.iloc[idx]["url"].tolist()
        imgs = rec_ta.iloc[idx]["image"].tolist()

        if len(urls) > 2:
            r = 3
        else:
            r = len(urls)

        return [(a, b) for a, b in zip(imgs, urls)][:r]

    def rank_popularity_recommend(self, input_clothes, rec_ta):

        features = ["rank", "sub-category", "cate", "url", "image"]
        rec_ta = rec_ta[features]

        rec_ta = rec_ta.sort_values(by="rank", ascending=False)
        print(rec_ta)
        idx = [a for a, b in enumerate(rec_ta["sub-category"].tolist()) if b == input_clothes]
        if len(idx) < 2:
            idx = [a for a, b in enumerate(rec_ta["cate"].tolist()) if b == input_clothes]

        urls = rec_ta.iloc[idx]["url"].tolist()
        imgs = rec_ta.iloc[idx]["image"].tolist()

        if len(urls) > 2:
            r = 3
        else:
            r = len(urls)

        return [(a, b) for a, b in zip(imgs, urls)][:r]
