from Lib.Lib_LangQwen import Qwen
from langchain.prompts import ChatPromptTemplate
import pyttsx3
import sys
import time


try:
    import sounddevice as sd
except ImportError as e:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)

import sherpa_ncnn


def create_recognizer():
    # Please replace the model files if needed.
    # See https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
    # for download links.
    recognizer = sherpa_ncnn.Recognizer(
        tokens="./models/sherpa-ncnn-conv-emformer-transducer-2022-12-06/tokens.txt",
        encoder_param="./models/sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.param",
        encoder_bin="./models/sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.bin",
        decoder_param="./models/sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.param",
        decoder_bin="./models/sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.bin",
        joiner_param="./models/sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.param",
        joiner_bin="./models/sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.bin",
        num_threads=4,
        decoding_method="modified_beam_search",
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,
    )
    return recognizer







engine = pyttsx3.init()


template = """你是一个智能助手。"""
human_template = "{text}"
chat_prompt = ChatPromptTemplate.from_messages([("system", template), ("human"), human_template])
chain = chat_prompt | Qwen()


while True:
    sure = input("你要说话吗？y/n:")
    if sure == "y":
        recognizer = create_recognizer()
        sample_rate = recognizer.sample_rate
        samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
        last_result = ""
        segment_id = 0
        with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
            while True:
                end = time.time()
                try:
                    sure = end-start
                    if sure>2:
                        print("")
                        del start
                        break
                except:
                    pass

                samples, _ = s.read(samples_per_read)  # a blocking read
                samples = samples.reshape(-1)
                # print(samples)
                recognizer.accept_waveform(sample_rate, samples)
                is_endpoint = recognizer.is_endpoint
                result = recognizer.text
                if result and (last_result != result):
                    temp = result[len(last_result):]
                    last_result = result
                    # print(f"{segment_id}: {temp}")
                    print(temp, end = "")
                    start = time.time()
                if result and is_endpoint:
                    segment_id += 1

        print("正在回答问题：", last_result)
        answer = chain.invoke({last_result})
        last_result = ""
        del result
        del temp
        print(answer)
        engine.say("        "+answer)
        engine.runAndWait()

    else:
        pass






