# gary-andreessen bot 🤖

## what is this?
gary-andreessen is my version of an agent, i guess. it's like a 14 yr old version of marc andreessen that talks mostly in audio-visual memes.

it began as a discord bot that can now also upload video and post to twitter.

it's very absurd. 

this bot now replies to comments on its own tweets, finds and replies to pmarca tweets, and will respond to a mention + youtube url with the timestamp attached so that you can get audio-video generations back from any youtube clip you want. you can then reply 'continue' and the bot will extend the video it just made for you.

https://x.com/thepatch_gary

## how does it work?
it works in combination with the ['concurrent_gary' docker-compose](https://github.com/betweentwomidnights/gary-backend-combined). mention the bot and it uses the llama to go search yt for a clip of marc andreessen speaking, then uses a 6 second clip to generate an audio continuation with musicgen and a video continuation on the node side.

it uses a teeny whisper model in combination with the llama to generate tweet text using the marc clip it found as part of its strange cocktail.

TODO: 

combine the 3 twitter handlers into one cohesive service.

clean up `llama3_cleaned.py` further by moving some more functions to a new python script. 
also clean it up by centralizing some of its system prompts.

incorporate with the eliza repository possibly to solve alot of the system prompting needed in each part of its pipeline.

use the transcript from the input audio to generate captions in the video i suppose.

replace any and all hardcoded text with llama outputs so that the 14 year old inside it can fully express itself while creating these trash videos.

complexify video generation fx and possibly add generative video.

dockerize this mfer so it can live in the cloud.

## commands in discord
- `!generate` - can be used with a youtube url + timestamp to iterate on whatever youtube clip you want

@gary continue (extends the audio and video output)

## current status
there's still many edge cases and i gotta make the llama be a little smarter about the marc andreessen timestamp it pulls. no one wants to hear lex fridman ask a question as the input audio/video prompt.

## technical details
- currently using `Llama-3.1-8B-Lexi-Uncensored_V2_F16` (downloaded from huggingface)
- sqlite db for handling past generations on twitter.
- fastapi server for the llama
- agent-twitter-client Scraper for some functionality, although most of what's here is just custom things.

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
