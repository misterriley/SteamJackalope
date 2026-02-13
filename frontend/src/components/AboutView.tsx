import React from 'react';
import MarkdownView from './MarkdownView';

const ABOUT_MD = String.raw`# Hybrid Steam Game Recommendations

Welcome to my project! I've been building variations on this tool for around a decade and finally have it in a form I feel comfortable releasing to the public. I love video games and statistical modeling. Building my own video game recommendation system is at the intersection of these pursuits and naturally I got [nerdsniped](https://xkcd.com/356/) by it.

The goal of this engine is to generate recommendations for Steam games based on a combination of factors that all influence how people respond to the games they play. Currently the options are semantic similarity, tag similarity, game quality, "discovery", popularity, release date, difficulty, and length. Semantic similarity is measured using latent vector representations of text, tag similarity is measured with a different vector built from user generated tags (tag vectors), and quality is determined by the number of positive and negative reviews that players have left on Steam. Discovery is a setting that affects the way that quality is calculated, by assigning high or low trust to small sets of user reviews. The available data introduce some problems that have to be worked around to get useful signals, and the way that we tackle those problems is laid out here. 

Users of this site can input a natural language prompt and a list of games that function as "seeds" for a search. Prompts can be descriptions of content ("shooter set in space"), emotions ("melancholy"), playstyle ("zenlike"), or any description that can help the user describe what they're trying to find. From these entries, the website tries to match the meanings of text and tags on any seeds to games in the database. Users can weight these scores, along with other values such as difficulty and popularity, as they like, which changes the sorting.   

## Other Recommendation Engines

There are several other very good recommendation engines, and if you are interested in finding new games on Steam then I suggest you shop around and find the one that best suits your needs. Here are a few other engines and what I think they excel at:

- [steampeek.hu](https://steampeek.hu/): Intuitive ranking of games based on similarity, somehow always comes up with the "right" answers. The developer clearly has a passion for the project. Steampeek is the largest inspiration for how this site works. If you're going to use one recommendation site besides this one, I would suggest you use Steampeek.
- [srec.ai](https://srec.ai/): A relatively new engine that has tag-based and smart recommender modes. It appears there is some fancy neural network training that happens behind the scenes. 
- [Steam Recommender](https://store.steampowered.com/recommender): Steam's own recommendation engine. Allows preferences for popular vs. niche games and older vs. newer games. Bases recommendations on your Steam play history, and has access to millions of data points for building up recommendations through collaborative filtering. 
- [Quantic Foundry](https://apps.quanticfoundry.com/surveys/start/gamerprofile/): Get recommendations based on your gamer personality. It's like Zodiac signs for the socially awkward. 

While they aren't technically recommendation engines, these websites are resources to find games based on searches and sorts: 

- [steam250.com](https://steam250.com/): Polished UI, metrics for people who really want to geek out, and endless ways to slice up data. They use what I would consider very low discovery for their rankings (except for the hidden gems section). Compare their top 250 list with the ranking on this site with Discovery set to "Known Quantities."
- [Metacritic (PC)](https://www.metacritic.com/browse/game/pc/): Includes non-Steam games, aggregates critic viewpoints instead of users who get mad about things that don't matter to everyone.
- [OpenCritic](https://opencritic.com/): Similar to metacritic, but is more open about their methodology. Sometimes the scores listed on the two websites differ dramatically.
- [IsThereAnyDeal](https://isthereanydeal.com): Finds the best deal on a game you like, and gives information about what deals existed in the past. 
- [GameFAQs](https://gamefaqs.gamespot.com/): Contains games from pretty much every platform you can imagine. Allows users to say how difficult a game is and estimate the number of hours it takes to beat it. Hosts good guides for many games. 
- [SteamSpy](https://steamspy.com/): Has some unique statistics that it estimates, like the number of people who own the game and how long people have played it. 
- [SteamDB](https://steamdb.info/): Serves an ocean of data about every steam game. Describing everything it does would take pages. 

Please note that if a website is listed here that does not imply that I am associated with them in anyway, nor does it mean that they endorse this website. Similarly, this website has no affiliation with Steam or Valve. All information about Steam games was obtained through public APIs in accordance with the terms of use.`;

const AboutView: React.FC = () => {
  return <MarkdownView content={ABOUT_MD} />;
};

export default AboutView;
