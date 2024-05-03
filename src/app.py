import base64
import json
import os
from urllib.parse import parse_qs

import boto3
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_aws import BedrockLLM
from langchain_community.retrievers import AmazonKendraRetriever

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


def lambda_handler(event, context):

    # リクエストボディを取得する
    if event["isBase64Encoded"]:
        # デコードし、さらにクエリストリングをパースする
        body = base64.b64decode(event["body"]).decode("utf-8")
        request_body = parse_qs(body)
    else:
        request_body = event["body"]

    # プロンプトを作成する
    prompt_template_qa = """
    あなたは経験豊富なソフトウェアエンジニアです。
    以下のコードはリーダブルコードの原則に基づいてレビューを行いたいと考えています。
    提供されたコードを精査し、改善点を具体的に指摘してください。
    また、コードの可読性を高めるための具体的な提案も行ってください。
    レビュー結果は日本語でお答えください。

    --- コード開始 ---
    {code}
    --- コード終了 ---

    1. コードの可読性に関する全体的な評価はどうですか？
    2. どのような部分が特に改善が必要ですか？その理由も含めて説明してください。
    3. 可読性を向上させるために具体的な変更を提案してください。
    """
    prompt_qa = PromptTemplate(template=prompt_template_qa, input_variables=["code"])
    formatted_question = prompt_qa.format(code=request_body)
    chain_type_kwargs = {"prompt": prompt_qa}

    try:
        # チェーンを実行する
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs=chain_type_kwargs)
        result = chain.invoke(formatted_question)
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "Error",
                    "result": e,
                }
            ),
        }

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Success", "result": result}),
    }
