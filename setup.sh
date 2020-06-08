mkdir -p ~/.flask/

echo "\
[server]\n\
headless = true|n|
enableCORS = false\n\
port = 5000\n\
" > ~/.flask/config.toml
