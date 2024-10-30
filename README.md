# gary-andreessen bot 🤖

## what is this?
this is the beginning of the gary-andreessen bot. it should be posting to twitter shortly but right now it's for discord.

## how does it work?
it works in combination with the ['concurrent_gary' docker-compose](https://github.com/betweentwomidnights/gary-backend-combined) that is located in this repo. mention the bot and it uses the llama to go search yt for a clip of marc andreessen speaking, then uses it to make an absurd video.

## commands
- `!generate` - can be used with a youtube url + timestamp to iterate on whatever youtube clip you want

## current status
there's still many edge cases and i gotta make the llama be a little smarter about the marc andreessen timestamp it pulls. no one wants to hear lex fridman ask a question as the input prompt.

## technical details
- currently using `llama-2-7b-chat.Q4_K_M.gguf` (downloaded from huggingface)
- probable upgrade to llama 3.2 coming soon

## origins
this bot was built for ai16z as a personal hackathon.

### find us here
- [ai16z discord](https://discord.gg/ai16z)
- [birthplace discord](https://discord.gg/VECkyXEnAd)
- [the collabage patch](https://thecollabagepatch.com) - see gary's many forms

## credits
this is a collab between:
- me
- gpt
- claude
- [@veryvanya](https://x.com/veryvanya)'s fine-tune

the use case emerged out of the ether.
