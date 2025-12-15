from datasets import load_dataset


def get_dialogue_sentences(lang, target_count=500000):
    sentences = set()  # Use set to avoid duplicates
    print(f"📥 Loading Dialogue Data for {lang}...")

    # SOURCE 1: Tatoeba (Clean, conversational, short sentences)
    # Great for comic bubbles like "Where are you going?"
    try:
        # Load English-Target pair
        # Just a dummy for en
        pair_name = f"en-{lang}" if lang != "en" else "en-fr"
        tatoeba = load_dataset("tatoeba", lang1="en",
                               lang2=lang, split="train")

        print(f"   - Scanning Tatoeba...")
        for item in tatoeba:
            text = item['translation'][lang]
            if 2 < len(text) < 40:  # Comic bubble length
                sentences.add(text)

        print(f"   => Got {len(sentences)} from Tatoeba")
    except Exception as e:
        print(f"   (Skipping Tatoeba: {e})")

    # SOURCE 2: OPUS-100 (Massive, mixed sources including subtitles)
    if len(sentences) < target_count:
        try:
            print(f"   - Scanning OPUS-100...")
            # 'en-ko' for Korean, 'en-zh' for Chinese
            opus_lang = "zh" if lang == "zh_cn" else lang
            opus = load_dataset("Helsinki-NLP/opus-100",
                                f"en-{opus_lang}", split="train", streaming=True)

            for i, item in enumerate(opus):
                text = item['translation'][opus_lang]
                if 2 < len(text) < 40:
                    sentences.add(text)

                if len(sentences) >= target_count:
                    break
            print(f"   => Total now {len(sentences)}")
        except Exception as e:
            print(f"   (Skipping OPUS-100: {e})")

    return list(sentences)


# --- 1. Fetch Korean ---
ko_sentences = get_dialogue_sentences("ko", target_count=500000)
print(f"✅ Final Korean Count: {len(ko_sentences)}")

# --- 2. Fetch Chinese ---
# Note: For OPUS, 'zh' usually covers simplified.
zh_sentences = get_dialogue_sentences("zh_cn", target_count=500000)
print(f"✅ Final Chinese Count: {len(zh_sentences)}")

# --- 3. Save ---
with open("comic_sentences_ko.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(ko_sentences))

with open("comic_sentences_zh.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(zh_sentences))
