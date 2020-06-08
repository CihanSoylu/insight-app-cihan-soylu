mkdir -p ~/.flask/

echo "\
[server]\n\
headless = true|n|
enableCORS = false\n\
port = $PORT\n\
" > ~/.flask/config.toml
