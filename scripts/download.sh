#!/bin/bash
TOKEN=$2

download_disney() {
    echo "Download checkpoint for Disney..."
    wget https://civitai.com/api/download/models/69832\?token\=${TOKEN} -P ./models/Model --content-disposition --no-check-certificate
}

download_moxin () {
    echo "Download checkpoints for MoXin..."
    wget https://civitai.com/api/download/models/106289\?token\=${TOKEN} -P ./models/Model --content-disposition --no-check-certificate
    wget https://civitai.com/api/download/models/14856\?token\=${TOKEN} -P ./models/LoRA --content-disposition --no-check-certificate
}

download_pixart () {
    echo "Download checkpoint for PixArt..."
    wget https://civitai.com/api/download/models/220049\?token\=${TOKEN} -P ./models/Model --content-disposition --no-check-certificate
}

download_origami () {
    echo "Download checkpoints for origami..."
    wget https://civitai.com/api/download/models/270085\?token\=${TOKEN} -P ./models/Model --content-disposition --no-check-certificate
    wget https://civitai.com/api/download/models/266928\?token\=${TOKEN} -P ./models/LoRA --content-disposition --no-check-certificate
}

download_threeDelicacy () {
    echo "Download checkpoints for threeDelicacy..."
    wget https://civitai.com/api/download/models/36473\?token\=${TOKEN} -P ./models/Model --content-disposition --no-check-certificate
}

download_toonyou () {
    echo "Download checkpoint for Toonyou..."
    wget https://civitai.com/api/download/models/125771\?token\=${TOKEN} -P ./models/Model --content-disposition --no-check-certificate
}

download_zaum () {
    echo "Download checkpoints for Zaum..."
    wget https://civitai.com/api/download/models/428862\?token\=${TOKEN} -P ./models/Model --content-disposition --no-check-certificate
    wget https://civitai.com/api/download/models/18989\?token\=${TOKEN} -P ./models/LoRA --content-disposition --no-check-certificate
}

download_felted () {
    echo "Download checkpoints for Felted..."
    wget https://civitai.com/api/download/models/428862\?token\=${TOKEN} -P ./models/Model --content-disposition --no-check-certificate
    wget https://civitai.com/api/download/models/86739\?token\=${TOKEN} -P ./models/LoRA --content-disposition --no-check-certificate
}

if [ -z "$1" ]; then
    echo "Please input the model you want to download."
    echo "Supported model: all, disney, moxin, pixart, paperArt, threeDelicacy, toonyou, zaum."
    exit 1
fi

declare -A download_func=(
    ["disney"]="download_disney"
    ["moxin"]="download_moxin"
    ["pixart"]="download_pixart"
    ["origami"]="download_origami"
    ["threeDelicacy"]="download_threeDelicacy"
    ["toonyou"]="download_toonyou"
    ["zaum"]="download_zaum"
    ["felted"]="download_felted"
)

execute_function() {
    local key="$1"
    if [[ -n "${download_func[$key]}" ]]; then
        ${download_func[$key]}
    else
        echo "Function not found for key: $key"
    fi
}


for arg in "$@"; do
    case "$arg" in
        disney|moxin|pixart|origami|threeDelicacy|toonyou|zaum|felted)
            model_name="$arg"
            execute_function "$model_name"
            ;;
        all)
            for model_name in "${!download_func[@]}"; do
                execute_function "$model_name"
            done
            ;;
        *)
            echo "Invalid argument: $arg."
            exit 1
            ;;
    esac
done
