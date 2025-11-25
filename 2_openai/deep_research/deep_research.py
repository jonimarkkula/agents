from agents import Runner, gen_trace_id, trace
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI
from research_manager import research_agent

client = OpenAI()

load_dotenv(override=True)

async def run(query: str):
    trace_id = gen_trace_id()
    with trace("Research trace", trace_id=trace_id):
        yield "Research started… please wait."
        try:
            stream = Runner.run_streamed(research_agent, query)
        except Exception as exc:
            yield f"Research failed: {exc}"
            return

        output_text = ""
        status_updates: list[str] = []

        def format_partial_output() -> str:
            base = output_text or ""
            if status_updates:
                base = f"{base}\n\n**Status updates**\n" + "\n".join(status_updates)
            return base or "Research in progress…"

        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                continue
            if event.type == "agent_updated_stream_event":
                status_updates.append(f"{event.new_agent.name} is working…")
                yield format_partial_output()
            elif event.type == "run_item_stream_event":
                item = event.item
                if item.type == "tool_call_item":
                    status_updates.append(f"Calling tool: {item.raw_item.name}")
                elif item.type == "tool_call_output_item":
                    status_updates.append("Tool call completed")
                elif item.type == "message_output_item":
                    if hasattr(item, "output") and item.output:
                        output_text += str(item.output)
                yield format_partial_output()

        final_output = getattr(stream, "final_output", None)
        if hasattr(final_output, "markdown_report"):
            yield final_output.markdown_report
        elif final_output is not None:
            yield str(final_output)
        else:
            yield format_partial_output()

with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# Deep Research")
    query_textbox = gr.Textbox(label="What topic would you like to research?")
    run_button = gr.Button("Run", variant="primary")
    report = gr.Markdown(label="Report")
    
    run_button.click(fn=run, inputs=query_textbox, outputs=report)
    query_textbox.submit(fn=run, inputs=query_textbox, outputs=report)

ui.launch(inbrowser=True)

