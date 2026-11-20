"""Streamlit UI for The Receptionist (revamped)."""
import streamlit as st
from receptionist.config import load_config
from receptionist.router import route_query
from receptionist.metrics import MetricsTracker
from receptionist.utils import count_tokens

st.set_page_config(page_title="The Receptionist", layout="wide")
config = load_config()

# Initialize session state
if 'messages' not in st.session_state:
	st.session_state.messages = []  # each: {role:'user'|'assistant', 'content':str, 'model':'local|cloud', 'label':str}
if 'metrics' not in st.session_state:
	st.session_state.metrics = MetricsTracker(config.get('pricing', {}), config.get('carbon_proxy', {}).get('enabled', True))
if 'last_classification' not in st.session_state:
	st.session_state.last_classification = None

st.title("The Receptionist")
st.caption("Cost-aware AI routing between local and cloud models.")

with st.sidebar:
	st.header("Dashboard")
	dash = st.session_state.metrics.get_metrics()
	st.metric("Local Tokens", dash['local_tokens'])
	st.metric("Cloud Tokens", dash['cloud_tokens'])
	st.metric("$ Baseline (all-cloud)", f"${dash['baseline_cost']:.3f}")
	st.metric("$ Actual", f"${dash['actual_cost']:.3f}")
	st.metric("$ Saved", f"${dash['savings']:.3f}")
	st.metric("Carbon Proxy", f"{dash['carbon_proxy']:.4f}")
	st.divider()
	routing_mode = st.radio("Routing mode", ("Local Only", "Auto"), help="Auto may send your current prompt to the configured cloud provider.")
	st.checkbox("Send next prompt to cloud", key="send_to_cloud", help="Sends one prompt to the configured cloud provider, then returns to the selected mode.")
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
		st.markdown(f"{badge}  \n{m['content']}")

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
	model_type, resp_obj, label, cls = route_query(
		user_input, config, escalate=manual_escalate, stream=streaming,
		local_only=routing_mode == "Local Only" and not manual_escalate,
	)
	# Assistant message container
	with st.chat_message('assistant', avatar=("☁️" if model_type=='cloud' else "💻")):
		badge = f"**{label}**"
		if streaming and hasattr(resp_obj, '__iter__') and not isinstance(resp_obj, str):
			placeholder = st.empty()
			collected = ''
			for chunk in resp_obj:
				collected += chunk
				placeholder.markdown(f"{badge}\n\n{collected}")
			response_text = collected.strip()
		else:
			response_text = resp_obj if isinstance(resp_obj, str) else ''.join(list(resp_obj))
			st.markdown(f"{badge}\n\n{response_text}")
		with st.expander("Routing Details", expanded=False):
			st.json(cls)
	# Metrics
	tokens_used = count_tokens(user_input + response_text)
	model_key = f"{config.get(model_type+'_model',{}).get('provider','unknown')}/{config.get(model_type+'_model',{}).get('name','unknown')}"
	st.session_state.metrics.record(user_input, response_text, model_type, tokens_used, model_key)
	st.session_state.messages.append({'role':'assistant','content':response_text,'model':model_type,'label':label,'classification':cls})
	st.session_state.last_classification = cls
	st.rerun()
