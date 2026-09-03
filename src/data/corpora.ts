/**
 * Built-in corpora.
 *
 * These are written, not borrowed. A word-level model with ~60k parameters
 * learns from repetition and fixed phrasing, so each corpus is composed to
 * repeat: shared refrains, a small vocabulary, and predictable sentence frames.
 * Prose with a large vocabulary would give a first-time visitor nothing but
 * mush, which reads as "this is broken" rather than "this model is small".
 */

export interface Corpus {
  id: string
  name: string
  blurb: string
  text: string
}

const LITTLE_LAMP = `the little lamp is on the hill
the little lamp is burning still
and all the town can see the light
the little lamp is burning bright

the little lamp is on the hill
the little lamp is burning still
the wind can blow the wind can cry
the little lamp will not go by

the little boat is on the sea
the little boat is sailing free
and all the town can see the sail
the little boat will never fail

the little boat is on the sea
the little boat is sailing free
the wind can blow the wind can cry
the little boat will not go by

the little bird is in the tree
the little bird is singing me
and all the town can hear the song
the little bird has sung it long

the little bird is in the tree
the little bird is singing me
the wind can blow the wind can cry
the little bird will not go by

so here is to the lamp and light
so here is to the burning bright
so here is to the boat and sea
so here is to the bird and tree`

const KITCHEN = `heat the pan and add the oil
add the onion and let it soften
add the garlic and stir for a minute
add the tomato and let it cook down
season with salt and pepper
serve with bread

heat the pot and add the water
add the salt and bring it to a boil
add the pasta and stir for a minute
drain the pasta and keep a cup of water
season with salt and pepper
serve with cheese

heat the oven and butter the tin
add the flour and the sugar to a bowl
add the eggs and stir until smooth
pour the batter into the tin
bake for thirty minutes until golden
serve with cream

heat the pan and add the butter
add the mushroom and let it brown
add the thyme and stir for a minute
add the cream and let it thicken
season with salt and pepper
serve with rice

chop the onion and chop the carrot
chop the celery and chop the garlic
heat the oil and add them to the pot
cook until soft and stir often
add the stock and bring it to a boil
simmer for twenty minutes and serve`

const FOX_STORY = `the fox went out on monday and the fox was very hungry
the fox found a berry by the river and the fox was happy
the fox went home and slept until morning

the fox went out on tuesday and the fox was very hungry
the fox found an apple by the wall and the fox was happy
the fox went home and slept until morning

the fox went out on wednesday and the fox was very hungry
the fox found a fish by the river and the fox was happy
the fox went home and slept until morning

the fox went out on thursday and the fox was very hungry
the fox found nothing by the river and the fox was sad
the fox went home and slept until morning

the fox went out on friday and the fox was very hungry
the fox found a berry and an apple and a fish
the fox shared the fish with the bird by the wall
the bird sang for the fox until morning

the fox went out on saturday and the fox was not hungry
the fox sat by the river with the bird
and the fox was happy until morning`

const WEATHER_LOG = `monday the sky was grey and the wind came from the north
tuesday the sky was clear and the wind came from the west
wednesday the sky was grey and the rain fell all afternoon
thursday the sky was clear and the sun was warm at noon
friday the sky was grey and the rain fell all morning
saturday the sky was clear and the wind came from the south
sunday the sky was grey and the fog sat on the water

monday the tide was high at dawn and low at dusk
tuesday the tide was low at dawn and high at dusk
wednesday the tide was high at noon and the boats stayed in
thursday the tide was low at noon and the boats went out
friday the tide was high at dawn and the boats went out
saturday the tide was low at dusk and the boats stayed in
sunday the tide was high at noon and the water was still

the wind came from the north and the water was rough
the wind came from the west and the water was calm
the wind came from the south and the air was warm
the fog sat on the water and the boats stayed in`

export const CORPORA: Corpus[] = [
  {
    id: 'lamp',
    name: 'Little lamp',
    blurb: 'Verse with heavy refrains. The smallest vocabulary, so it learns fastest.',
    text: LITTLE_LAMP,
  },
  {
    id: 'fox',
    name: 'The fox',
    blurb: 'A story on a weekly template. Watch it learn the days of the week.',
    text: FOX_STORY,
  },
  {
    id: 'kitchen',
    name: 'Kitchen steps',
    blurb: 'Instructions. Rigid phrasing, so the structure shows up early.',
    text: KITCHEN,
  },
  {
    id: 'weather',
    name: 'Weather log',
    blurb: 'Records with two interleaved patterns. Harder: it must track which one it is in.',
    text: WEATHER_LOG,
  },
]

export const DEFAULT_CORPUS = CORPORA[0]
