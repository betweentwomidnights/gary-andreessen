# gary-andreessen bot 🤖

## what is this?
this is the beginning of the gary-andreessen bot. 

it's a discord bot that can also upload video and post to twitter in a very specific manner.

## how does it work?
it works in combination with the ['concurrent_gary' docker-compose](https://github.com/betweentwomidnights/gary-backend-combined). mention the bot and it uses the llama to go search yt for a clip of marc andreessen speaking, then uses a 6 second clip to generate an audio continuation with musicgen and a video continuation on the node side.

it uses a teeny whisper model in combination with the llama to generate tweet text.

TODO: 

get it responding to comments on twitter.

get it scraping twitter for mentions of marc andreessen and post a reply.

use the transcript from the input audio to generate captions in the video i suppose.

POSSIBLE TODO:

use a more powerful model with a much larger context window and mega gpu (with the vector store stuff and postgres i think?) to make its video an actual relevant response to what is said to it, rather than just finding a random piece of marc talking.

## commands
- `!generate` - can be used with a youtube url + timestamp to iterate on whatever youtube clip you want

@gary continue (extends the audio and video output)

## current status
there's still many edge cases and i gotta make the llama be a little smarter about the marc andreessen timestamp it pulls. no one wants to hear lex fridman ask a question as the input audio/video prompt.

## technical details
- currently using `llama-2-7b-chat.Q4_K_M.gguf` (downloaded from huggingface)
- probable upgrade to llama 3.2 8b.

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
