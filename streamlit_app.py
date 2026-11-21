"""Streamlit UI for The Receptionist (revamped)."""
from time import perf_counter

import streamlit as st
from receptionist.config import load_config
from receptionist.router import route_query
from receptionist.models.errors import ProviderError
from receptionist.metrics import MetricsTracker
from receptionist.utils import count_tokens

st.set_page_config(page_title="The Receptionist", layout="wide")
config = dict(load_config())
cloud_budget_usd = config.get("cloud_budget_usd")

# Initialize session state
if 'messages' not in st.session_state:
	st.session_state.messages = []  # each: {role:'user'|'assistant', 'content':str, 'model':'local|cloud', 'label':str}
if 'last_classification' not in st.session_state:
	st.session_state.last_classification = None

def usd(value, decimals=4):
	return f"${value:.{decimals}f}" if value is not None else "Unavailable"


ROUTING_EXPLANATIONS = {
	"local only": "Local Only was selected.",
	"manual override": "You chose to send this prompt to the cloud.",
	"safety escalation": "Auto matched a keyword that triggers cloud routing.",
	"greeting / acknowledgment": "Auto recognized a greeting or acknowledgment.",
	"length override": "Auto sent the prompt to the cloud because it exceeded the local word limit.",
	"score >= boundary": "The Auto routing score met the cloud threshold.",
	"score < boundary": "The Auto routing score was below the cloud threshold.",
	"session cloud budget reached": "The session cloud budget was reached, so the prompt went local.",
	"cloud pricing unavailable": "Cloud pricing was unavailable, so the budget guard kept the prompt local.",
}


def show_response_details(message):
	classification = message.get("classification")
	if not classification:
		return
	reason = classification["reason"]
	st.caption(ROUTING_EXPLANATIONS.get(reason, reason))
	details = message.get("details")
	if details:
		cost = usd(details["cloud_cost"], decimals=6)
		st.caption(f"Provider: {details['provider']} · Response time: {details['seconds']:.2f} s · Estimated cloud cost: {cost}")
	with st.expander("Routing details"):
		st.json(classification)


st.title("The Receptionist")
st.caption("Each reply uses only the current prompt. Cloud requests also include the configured system instructions. Earlier messages stay in this session.")

with st.sidebar:
	st.header("Cloud model")
	providers = {"openai": "OpenAI", "anthropic": "Anthropic", "google": "Gemini"}
	default_models = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5", "google": "gemini-2.5-flash"}
	configured = config.get("cloud_model", {})
	configured_provider = configured.get("provider", "openai")
	provider = st.selectbox(
		"Cloud provider", tuple(providers), format_func=providers.get,
		index=list(providers).index(configured_provider) if configured_provider in providers else 0,
	)
	default_model = (configured.get("name") if configured_provider == provider else None) or default_models[provider]
	model = st.text_input("Model ID", value=default_model, key=f"model_{provider}").strip() or default_model
	with st.expander("Bring your own API keys", expanded=True):
		st.caption("Keys entered here are kept in this session. Leave a field blank to use configured credentials.")
		keys = {
			name: st.text_input(f"{label} API key", type="password", key=f"byok_{name}").strip()
			for name, label in providers.items()
		}
	cloud_model = {"provider": provider, "name": model}
	if keys[provider]:
		cloud_model["api_key"] = keys[provider]
	elif provider == configured_provider and configured.get("api_key"):
		cloud_model["api_key"] = configured["api_key"]
	config["cloud_model"] = cloud_model
	cloud_model_key = f"{provider}/{model}"
	if not isinstance(st.session_state.get("usage_metrics"), MetricsTracker):
		st.session_state.usage_metrics = MetricsTracker(config.get("pricing", {}), cloud_model_key)
	metrics = st.session_state.usage_metrics
	metrics.baseline_model = cloud_model_key
	budget_reason = metrics.cloud_budget_reason(cloud_budget_usd, cloud_model_key)
	st.divider()
	st.header("Dashboard")
	dash = metrics.get_metrics()
	st.metric("Local Tokens", dash['local_tokens'])
	st.metric("Cloud Tokens", dash['cloud_tokens'])
	st.metric("Estimated all-cloud baseline", usd(dash['baseline_cost']))
	st.metric("Estimated cloud spend", usd(dash['actual_cost']))
	st.metric("Estimated cloud savings", usd(dash['savings']))
	st.metric("Local request share", f"{dash['local_request_share']:.0%}")
	st.caption(f"Input: {dash['input_tokens']} · Output: {dash['output_tokens']} · Estimated usage on {dash['estimated_requests']} requests")
	st.caption(f"All-cloud baseline model: {cloud_model_key}")
	if cloud_budget_usd is not None:
		st.caption(f"Session cloud budget: {usd(cloud_budget_usd)}")
	if budget_reason:
		st.warning(f"Cloud requests will run locally: {budget_reason}.")
	st.divider()
	routing_mode = st.radio("Routing mode", ("Local Only", "Auto"), help="Auto can send this prompt and the configured system instructions to the cloud provider.")
	st.checkbox("Send next prompt to cloud", key="send_to_cloud", help="Sends your next prompt and the configured system instructions to the cloud provider, then returns to the selected mode.")
	streaming = st.checkbox("Stream Responses", value=True)
	st.write("Model Config:")
	st.code(f"Local: {config.get('local_model',{}).get('name')}\nCloud: {config.get('cloud_model',{}).get('name')}")
	st.divider()
	st.subheader("Latest Routing Detail")
	if st.session_state.last_classification:
		# Only show selective fields to keep concise
		cls = st.session_state.last_classification
		condensed = {
			'decision': cls.get('decision'),
			'confidence': cls.get('confidence'),
			'escalate_score': cls.get('escalate_score'),
			'reason': cls.get('reason'),
			'features': {
				'word_count': cls['features'].get('word_count'),
				'has_code': cls['features'].get('has_code'),
				'complex_density': cls['features'].get('complex_density'),
				'is_question': cls['features'].get('is_question'),
				'components': cls['features'].get('components')
			}
		}
		st.json(condensed)
	else:
		st.caption("No classification yet.")

# Render previous messages
for m in st.session_state.messages:
	with st.chat_message(m['role'], avatar="🧑" if m['role']=='user' else ("☁️" if m['model']=='cloud' else "💻")):
		badge = f"`{m['label']}`"
		st.markdown(badge)
		if m['content']:
			st.markdown(m['content'])
		if m.get('error'):
			status = "Partial response" if m['content'].strip() else "Request failed"
			st.error(f"{status}: {m['error']}")
		show_response_details(m)

def consume_cloud_override():
	st.session_state.cloud_override = st.session_state.send_to_cloud
	st.session_state.send_to_cloud = False


user_input = st.chat_input("Ask something...", on_submit=consume_cloud_override)
if user_input:
	st.session_state.messages.append({'role':'user','content':user_input,'model':'user','label':'User'})
	with st.chat_message('user', avatar="🧑"):
		st.markdown(user_input)
	# Route
	manual_escalate = st.session_state.pop("cloud_override", False)
	usage = []

	def collect_usage(input_tokens, output_tokens):
		usage[:] = [input_tokens, output_tokens]

	response_started = perf_counter()
	model_type, resp_obj, label, cls = route_query(
		user_input, config, escalate=manual_escalate, stream=streaming,
		local_only=routing_mode == "Local Only" and not manual_escalate,
		cloud_budget_reason=budget_reason, on_usage=collect_usage,
	)
	# Assistant message container
	with st.chat_message('assistant', avatar=("☁️" if model_type=='cloud' else "💻")):
		st.markdown(f"**{label}**")
		response_text = ""
		failure = None
		try:
			if isinstance(resp_obj, ProviderError):
				raise resp_obj
			if isinstance(resp_obj, str):
				response_text = resp_obj
				st.markdown(response_text)
			else:
				placeholder = st.empty()
				for chunk in resp_obj:
					response_text += chunk
					placeholder.markdown(response_text)
		except ProviderError as error:
			if error.partial:
				response_text = error.partial
				st.markdown(response_text)
			failure = str(error)
			status = "Partial response" if response_text.strip() else "Request failed"
			st.error(f"{status}: {failure}")
	elapsed = perf_counter() - response_started
	# Metrics
	model_key = cloud_model_key if model_type == "cloud" else f"{config.get('local_model', {}).get('provider', 'ollama')}/{config.get('local_model', {}).get('name', 'gemma2:2b')}"
	system_prompt = (config.get("system_prompt") or "") if model_type == "cloud" else ""
	input_tokens, output_tokens = usage or [
		count_tokens(user_input) + count_tokens(system_prompt), count_tokens(response_text),
	]
	cloud_cost = 0 if model_type == "local" else None
	if failure is None or usage or response_text.strip():
		metrics.record(model_type, model_key, input_tokens, output_tokens, estimated=not usage)
		if model_type == "cloud":
			cloud_cost = metrics.estimate_cost(model_key, input_tokens, output_tokens)
	st.session_state.messages.append({
		"role": "assistant", "content": response_text, "model": model_type,
		"label": label, "classification": cls, "error": failure,
		"details": {
			"provider": model_key.split("/", 1)[0], "seconds": elapsed,
			"cloud_cost": cloud_cost,
		},
	})
	st.session_state.last_classification = cls
	st.rerun()
