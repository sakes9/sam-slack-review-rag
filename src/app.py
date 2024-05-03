import base64
import json
import os
import re

import boto3
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_aws import BedrockLLM
from langchain_community.retrievers import AmazonKendraRetriever
from slack_bolt import App

# Kendraの検索エンジンを作成
KENDRA_INDEX_ID = os.getenv("KENDRA_INDEX_ID")  # KendraインデックスIDを取得する
ATTRIBUTE_FILTER = {
    "EqualsTo": {"Key": "_language_code", "Value": {"StringValue": "ja"}}
}  # 日本語のドキュメントのみを検索する
retriever = AmazonKendraRetriever(
    index_id=KENDRA_INDEX_ID, attribute_filter=ATTRIBUTE_FILTER, region_name="ap-northeast-1"
)

# BedrockLLMを作成する
client = boto3.client("bedrock-runtime", region_name="ap-northeast-1")
llm = BedrockLLM(
    client=client,
    model_id="anthropic.claude-v2:1",
    model_kwargs={"max_tokens_to_sample": 2000},
)

# Slackアプリを作成する
slack_app = App(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    process_before_response=True,  # リスナー関数での処理が完了するまで HTTP レスポンスの送信を遅延させる、3秒以内にリスナーのすべての処理が完了しなかった場合でも Slack 上でタイムアウトのエラー表示をさせない。
)


def lambda_handler(event, context):

    try:
        # Slackイベントを取得する
        headers = event.get("headers", {})
        if event["isBase64Encoded"]:
            body = base64.b64decode(event["body"])
        else:
            body = event["body"]

        # 辞書形式に変換する
        if not isinstance(body, dict):
            slack_event_json = json.loads(body)
        else:
            slack_event_json = body

        # Slack API の Event Subscriptions を有効化する際のイベントの場合
        if "challenge" in slack_event_json:
            challenge = slack_event_json["challenge"]
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"challenge": challenge}),
            }

        # 再送の場合はリクエストを無視する
        if "x-slack-retry-num" in headers:
            print("Detected x-slack-retry-num. Exiting to avoid processing a retry from Slack.")
            return {"statusCode": 200, "body": json.dumps({"message": "再送リクエストは無視します。"})}

        # Slackからのイベントの場合
        if "event" in slack_event_json:
            slack_event = slack_event_json["event"]

            # アプリへのメンションが付与されたメッセージの場合
            if slack_event["type"] == "app_mention":
                channel = slack_event["channel"]
                thread_ts = slack_event.get("thread_ts", slack_event["ts"])
                text = slack_event["text"]
                text_without_mention = re.sub(r"^<@(.+?)>", "", text).strip()

                # 自動応答メッセージをSlackに返す
                slack_app.client.chat_postMessage(
                    channel=channel,
                    thread_ts=thread_ts,
                    text="レビュー中・・・",
                )

                # プロンプトを作成する
                prompt_template_qa = """
                あなたはコードレビュアーです。以下のレビュー観点からコードのレビューを行ってください。

                レビュー観点:
                「第Ⅰ部 表面上の改善 2章. 名前に情報を詰め込む」の指示に従ってください。他の一般的なレビュー観点は考慮しないでください。

                指摘事項:
                1. 指摘事項を挙げてください。どの指示に基づくものかを具体的に明記してください。
                2. 各指摘点に対して、どのような改善が必要かとその理由を詳述してください。
                3. 必要な改善策を具体的に示し、改善案があれば改善後のコードを提供してください。

                {context}

                --- コード開始 ---
                {question}
                --- コード終了 ---

                レビュー結果は日本語でお答えください。
                """
                prompt_qa = PromptTemplate(template=prompt_template_qa, input_variables=["context", "question"])
                formatted_question = prompt_qa.format(context="", question=text_without_mention)
                chain_type_kwargs = {"prompt": prompt_qa}

                # チェーンを実行する
                chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs=chain_type_kwargs)
                response = chain.invoke(formatted_question)
                result = response["result"]

                # 応答結果をSlackに返す
                slack_app.client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=result)
    except Exception as e:
        # エラーメッセージをSlackに返す
        slack_app.client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text=f"レビュー中にエラーが発生しました。error: {str(e)}"
        )
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "Error",
                    "result": str(e),
                }
            ),
        }

    return {"statusCode": 200, "body": json.dumps({"message": "レビューが完了しました"})}
