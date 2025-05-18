import json
import logging
import os
from typing import Any

from .utils.model_utils import load_path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler as AwsLambdaSlackHandler
from slack_bolt.adapter.fastapi import SlackRequestHandler
# from slack_bolt.oauth.oauth_settings import OAuthSettings
# from slack_sdk.oauth.installation_store import FileInstallationStore
from slack_sdk.oauth.state_store import FileOAuthStateStore

from strands import Agent
from strands_tools import (
    agent_graph,
    calculator,
    editor,
    environment,
    generate_image,
    http_request,
    image_reader,
    journal,
    load_tool,
    nova_reels,
    python_repl,
    retrieve,
    shell,
    swarm,
    think,
    use_aws,
    use_llm,
    workflow,
)
from strands_agents_builder.handlers.callback_handler import CallbackHandler
from strands_agents_builder.utils.model_utils import load_model
from strands_agents_builder.utils.kb_utils import load_system_prompt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

os.environ["NO_COLOR"] = "1"
os.environ["STRANDS_TOOL_CONSOLE_MODE"] = "enabled"

# installation_store = FileInstallationStore(base_dir="./data/installations")
state_store = FileOAuthStateStore(
    expiration_seconds=600,
    base_dir="./data/states"
)

app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
    # process_before_response=True,
    # installation_store=installation_store,
    # oauth_settings=OAuthSettings(
    #     client_id=os.environ.get("SLACK_CLIENT_ID"),
    #     client_secret=os.environ.get("SLACK_CLIENT_SECRET"),
    #     scopes=["app_mentions:read", "chat:write", "commands"],
    #     installation_store=installation_store,
    #     state_store=state_store,
    # )
)

strands_agent = None


def get_strands_agent() -> Agent:
    global strands_agent, model_provider
    
    if strands_agent is None:
        model_provider = os.environ.get("STRANDS_MODEL_PROVIDER", "bedrock")
        model_config = os.environ.get("STRANDS_MODEL_CONFIG", "{}")
        model_path = load_path(model_provider)
        model = load_model(model_path, json.loads(model_config))
        system_prompt = load_system_prompt()
        
        strands_agent = Agent(
            model=model,
            system_prompt=system_prompt,
            callback_handler=CallbackHandler().callback_handler,
        )
        
        strands_agent.tools = [
        shell,
        editor,
        http_request,
        python_repl,
        calculator,
        retrieve,
        use_aws,
        load_tool,
        environment,
        use_llm,
        think,
        load_tool,
        journal,
        image_reader,
        generate_image,
        nova_reels,
        agent_graph,
        swarm,
        workflow,
        # store_in_kb,
        # strand,
        # welcome,
    ]
    
    logger.info(f"Loaded Strands agent with model provider: {model_provider}")
    
    return strands_agent

def get_user_display_name(client, user_id: str) -> str:
    try:
        if not user_id or user_id == "unknown":
            return "unknown"
            
        response = client.users_info(user=user_id)
        if response.get("ok"):
            user = response.get("user", {})
            return user.get("real_name") or user.get("name", user_id)
    except Exception as e:
        logger.warning(f"Failed to get user info for {user_id}: {str(e)}")
    return user_id

def build_conversation_history(client, channel_id: str, thread_ts: str) -> str:
    try:
        logger.info(f"Fetching conversation history for channel: {channel_id}, thread_ts: {thread_ts}")
        
        auth_response = client.auth_test()
        if not auth_response.get("ok"):
            logger.warning(f"Failed to get bot user ID: {auth_response.get('error', 'unknown error')}")
            return ""
            
        bot_user_id = auth_response.get("user_id")
        bot_display_name = get_user_display_name(client, bot_user_id)
        logger.info(f"Bot user ID: {bot_user_id}, Display name: {bot_display_name}")
        
        response = client.conversations_replies(
            channel=channel_id,
            ts=thread_ts,
            inclusive=True,
            limit=100
        )
        
        if not response.get("ok"):
            logger.warning(f"Failed to fetch thread messages: {response.get('error', 'unknown error')}")
            return ""
            
        messages = response.get("messages", [])
        conversation = []
        
        for msg in messages:
            user_id = msg.get("user", "unknown")
            display_name = get_user_display_name(client, user_id)
            text = msg.get("text", "")
            conversation.append(f"{display_name}: {text}")
        
        return "\n".join(conversation)
        
    except Exception as e:
        logger.error(f"Error building conversation history: {str(e)}", exc_info=True)
        return ""

@app.event("app_mention")
def handle_mentions(body: dict[str, Any], say, logger, client) -> None:
    try:
        event = body.get("event", {})
        channel = event.get("channel")
        thread_ts = event.get("thread_ts") or event.get("ts")
        
        logger.info(f"Event data: channel={channel}, thread_ts={thread_ts}")
        
        conversation = build_conversation_history(client, channel, thread_ts)
        agent = get_strands_agent()
        
        try:
            response = agent(conversation)
            response_text = str(response) if response else "申し訳ありませんが、回答を生成できませんでした。"
            say(response_text, thread_ts=thread_ts)
            logger.info(f"Responded to mention: {response_text}")
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            say(f"エラーが発生しました: {str(e)}", thread_ts=thread_ts)
            
    except Exception as e:
        logger.error(f"Error in handle_mentions: {str(e)}", exc_info=True)
        say("申し訳ありませんが、エラーが発生しました。後でもう一度お試しください。", thread_ts=thread_ts)


fastapi_app = FastAPI(title="Strands Slack API")
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

slack_handler = SlackRequestHandler(app)
aws_lambda_handler = AwsLambdaSlackHandler(app)

def lambda_handler(event, context):
    logger.info(f"Received event: {json.dumps(event, ensure_ascii=False)}")
    
    if "challenge" in event:
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"challenge": event["challenge"]}),
        }
    
    return aws_lambda_handler.handle(event, context)

@fastapi_app.post("/slack/events")
async def slack_events(request: Request):
    body = await request.body()
    body_dict = json.loads(body)
    
    if "challenge" in body_dict:
        return {"challenge": body_dict["challenge"]}
        
    return await slack_handler.handle(request)

@fastapi_app.get("/slack/install")
async def install(request: Request):
    return await slack_handler.handle(request)

@fastapi_app.get("/slack/oauth_redirect")
async def oauth_redirect(request: Request):
    return await slack_handler.handle(request)

@fastapi_app.get("/health")
async def health_check():
    return {"status": "ok"}

import click

@click.command()
@click.option("--port", "-p", default=3000, help="Port to run the server on")
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind the server to")
@click.option("--reload/--no-reload", default=True, help="Enable/disable auto-reload")
@click.option("--log-level", default="info", help="Log level (debug, info, warning, error, critical)")
def run_server(port: int, host: str, reload: bool, log_level: str) -> None:
    """Run the FastAPI server for Slack integration.
    
    Example:
        uv run strands-slack --port 3000 --host 0.0.0.0 --reload --log-level info
    """
    import uvicorn
    
    os.makedirs("./data/installations", exist_ok=True)
    os.makedirs("./data/states", exist_ok=True)
    
    logger.info(f"Starting FastAPI server on http://{host}:{port}")
    logger.info(f"API Docs: http://{host}:{port}/docs")
    
    uvicorn.run(
        "strands_agents_builder.slack:fastapi_app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level.lower(),
        factory=False,
    )

if __name__ == "__main__":
    run_server()
