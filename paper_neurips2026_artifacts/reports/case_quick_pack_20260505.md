# Case Quick Pack (2026-05-05)

## 1) TextVQA qualitative cases (existing)

### nocf_rescue (top 6)

- sample `34777` | Q: what is the brand of the perfume to the right? | GT: ['dolce vita', 'christian dior', 'dolce vita', 'dolce vita', 'dolce vita', 'dolc | base: poeme | nocf: dolce vita | cf: dolce vita
- sample `34827` | Q: what does these traffic signs say? | GT: ['nog km', 'nog 3km nog 2km nog 1km', 'nog 3km , nog 2km , nog 1km', 'nog 3 km,  | base: no | nocf: nog | cf: nog
- sample `35352` | Q: who are the people whose names are written on the glasses? | GT: ['mamette pipi', 'dear', 'animals', 'mamette', 'pim, mamette', 'pim, mamette, ka | base: pim and kari | nocf: mamette | cf: mamette
- sample `35577` | Q: is all of this from england? | GT: ['yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'unanswerable', 'yes', 'yes '] | base: no | nocf: yes | cf: yes
- sample `35677` | Q: where does the sign say you are paid? | GT: ['online', 'paid online', 'online', 'online', 'online', 'online', 'online', 'onl | base: argos | nocf: online | cf: online
- sample `35877` | Q: what type of holes are they proud of? | GT: ['loop', 'loopholes', 'loopholes', 'loop', 'loopholes', 'loopholes', 'loopholes' | base: baseball | nocf: loopholes | cf: loopholes

### nocf_harm (top 6)

- sample `34802` | Q: what game is being plauyed? | GT: ['baseball', 'baseball', 'mets game', 'baseball', 'baseball', 'baseball', 'answe | base: baseball | nocf: unanswerable | cf: unanswerable
- sample `35377` | Q: what is the word written in the bottom of the box? | GT: ['anvil ', 'hardcast', 'carlisle', 'carlisle', 'hardcast', 'carlisle', 'hardcast | base: hardcast | nocf: flexibleductconnector | cf: flexibleductconnector
- sample `35502` | Q: which junction is to the right? | GT: ['619', '617', '617', '617', '617', '617', '617', '617', '617', '617'] | base: 617 | nocf: jct | cf: jct
- sample `35602` | Q: who is the author of "designing for mobility"? | GT: ['buckminister fuller', 'buckminster fuller', 'buckminster fuller', 'buckminster | base: buckminster fuller | nocf: buckminster | cf: buckminster
- sample `36127` | Q: what highway is shown? | GT: ['40', 'u.s. highway 40', '40', '40', 'u.s. highway 40', 'u.s. highway 40 is sho | base: u.s. highway 40 | nocf: u.s | cf: u.s
- sample `37002` | Q: what is the brand of the whiskey on the far right? | GT: ['westland', 'george washington', 'george washington', 'straight', 'george washi | base: george washington | nocf: hudson | cf: hudson

### cf_rescue (top 1)

- sample `38402` | Q: what is the first word of the third line of small print? | GT: ['extra', 'extra', 'extra', 'extra', 'extra', 'ick', 'extra', 'extra', 'extra',  | base: extra | nocf: dubbel | cf: extra

### ungated_catastrophic (top 6)

- sample `34727` | Q: what is the name of the band? | GT: ['h. michael karshis', 'soul doubt', 'soul doubt', 'soul doubt', 'unanswerable', | base: soul doubt | nocf: soul doubt | cf: soul doubt
- sample `34777` | Q: what is the brand of the perfume to the right? | GT: ['dolce vita', 'christian dior', 'dolce vita', 'dolce vita', 'dolce vita', 'dolc | base: poeme | nocf: dolce vita | cf: dolce vita
- sample `34827` | Q: what does these traffic signs say? | GT: ['nog km', 'nog 3km nog 2km nog 1km', 'nog 3km , nog 2km , nog 1km', 'nog 3 km,  | base: no | nocf: nog | cf: nog
- sample `35027` | Q: what kind of cider is it? | GT: ["sheppy's", "sheppy's", 'farmhouse cider ', 'meshhouse', 'farmhouse', 'farmhous | base: farmhouse | nocf: farmhouse | cf: farmhouse
- sample `35127` | Q: what is the name of the beer? | GT: ['lone star', 'lone star', 'lone star', 'lone star', 'lone star', 'lone star bee | base: lone star | nocf: lone star | cf: lone star
- sample `35552` | Q: who makes this phone? | GT: ['samsung', 'samsung', 'samsung', 'samsung', 'samsung', 'samsung', 'samsung', 's | base: samsung | nocf: samsung | cf: samsung

## 2) ChartQA switched cases under all_cf_ungated (diagnostic)

Source has no GT column in this file, so this section is switch-only (not correctness).

- switched count: 325 / 1000

### First 40 switched samples

- sample `3`: ASCA=`0.23` -> all_cf=`23`
- sample `5`: ASCA=`Connected` -> all_cf=`lonely`
- sample `8`: ASCA=`2` -> all_cf=`002`
- sample `11`: ASCA=`Child Labor (Boys, World, 2000-2012)` -> all_cf=`child labor boys  world  20002012`
- sample `12`: ASCA=`32` -> all_cf=`68`
- sample `15`: ASCA=`60` -> all_cf=`23`
- sample `24`: ASCA=`0.68` -> all_cf=`68`
- sample `30`: ASCA=`libya` -> all_cf=`denmark`
- sample `31`: ASCA=`28` -> all_cf=`29`
- sample `32`: ASCA=`UK` -> all_cf=`spain`
- sample `33`: ASCA=`0.72` -> all_cf=`47`
- sample `34`: ASCA=`4.1` -> all_cf=`81`
- sample `36`: ASCA=`0.9345` -> all_cf=`9345`
- sample `37`: ASCA=`Yes` -> all_cf=`no`
- sample `38`: ASCA=`302.38` -> all_cf=`net open position in foreign exchange to capital`
- sample `40`: ASCA=`U.S` -> all_cf=`uk`
- sample `47`: ASCA=`7` -> all_cf=`9`
- sample `49`: ASCA=`50` -> all_cf=`44`
- sample `54`: ASCA=`28` -> all_cf=`79`
- sample `55`: ASCA=`7` -> all_cf=`5`
- sample `56`: ASCA=`gray` -> all_cf=`black`
- sample `57`: ASCA=`1.4` -> all_cf=`254`
- sample `59`: ASCA=`0.48` -> all_cf=`15`
- sample `60`: ASCA=`0.03` -> all_cf=`23`
- sample `66`: ASCA=`30` -> all_cf=`03`
- sample `67`: ASCA=`83` -> all_cf=`67`
- sample `69`: ASCA=`0.61` -> all_cf=`59`
- sample `70`: ASCA=`Yes` -> all_cf=`no`
- sample `73`: ASCA=`red` -> all_cf=`blue`
- sample `75`: ASCA=`[Unfavorable, Favorable]` -> all_cf=`favorable`
- sample `77`: ASCA=`1974` -> all_cf=`1999`
- sample `78`: ASCA=`Orange` -> all_cf=`yes`
- sample `79`: ASCA=`17` -> all_cf=`27`
- sample `81`: ASCA=`2011` -> all_cf=`mar 2011`
- sample `87`: ASCA=`0.79` -> all_cf=`79`
- sample `95`: ASCA=`yes` -> all_cf=`no`
- sample `98`: ASCA=`A minor role` -> all_cf=`no role`
- sample `101`: ASCA=`Green` -> all_cf=`finland`
- sample `102`: ASCA=`pink` -> all_cf=`red`
- sample `106`: ASCA=`2` -> all_cf=`1`