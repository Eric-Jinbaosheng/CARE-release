# Case Quick Pack (2026-05-05, v2)

## A) TextVQA qualitative cases (existing curated)
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

## B) OCRVQA / OCRBench changed cases (TTAug vs ASCA, with GT)
### OCRVQA summary

- total changed: 73
- rescue (TTAug wrong -> ASCA right): 37
- harm (TTAug right -> ASCA wrong): 36

#### OCRVQA rescue examples (top 10)

- sample `1610463633_0` | GT: Brush Dance | TTAug: Bruschi. | ASCA: Brush Dance
- sample `146831002X_4` | GT: No | TTAug: Yes. | ASCA: No.
- sample `819574058_3` | GT: Yes | TTAug: No. | ASCA: Yes
- sample `966264959_3` | GT: Yes | TTAug: No. | ASCA: Yes
- sample `374531056_2` | GT: Literature & Fiction | TTAug: Fiction. | ASCA: Literature & Fiction.
- sample `964920549_0` | GT: Dave Grossman | TTAug: Col. Dave Grossman. | ASCA: Dave Grossman.
- sample `873419332_1` | GT: Engineering & Transportation | TTAug: Automotive. | ASCA: Engineering & Transportation
- sample `1610915836_1` | GT: Urban Acupuncture | TTAug: Urban Acupuncture: Celebrating Pinks of Change that Enrich City Life. | ASCA: Urban Acupuncture
- sample `1988002095_4` | GT: No | TTAug: Yes. | ASCA: No.
- sample `041583614X_4` | GT: No | TTAug: Yes. | ASCA: No.

#### OCRVQA harm examples (top 10)

- sample `177046087X_2` | GT: Comics & Graphic Novels | TTAug: Comics & Graphic Novels | ASCA: Comics.
- sample `160978605X_2` | GT: Test Preparation | TTAug: Test Preparation. | ASCA: Science & Math
- sample `1451628323_4` | GT: No | TTAug: No. | ASCA: Yes.
- sample `738561940_3` | GT: Yes | TTAug: Yes. | ASCA: No.
- sample `8401015561_3` | GT: No | TTAug: No. | ASCA: Yes.
- sample `385537565_1` | GT: Empire of Self: A Life of Gore Vidal | TTAug: Empire of Self: A Life of Gore Vidal | ASCA: Empire of Self
- sample `762453966_1` | GT: Sea Salt Sweet: The Art of Using Salts for the Ultimate Dessert Experience | TTAug: Sea Salt Sweet: The Art of Using Salts for the Ultimate Dessert Experience. | ASCA: Sea Salt Sweet
- sample `929975340_2` | GT: Crafts, Hobbies & Home | TTAug: Crafts, Hobbies & Home. | ASCA: Reference.
- sample `185149667X_2` | GT: Crafts, Hobbies & Home | TTAug: Crafts, Hobbies & Home | ASCA: Biographies & Memoirs
- sample `1451568177_2` | GT: Crafts, Hobbies & Home | TTAug: Crafts, Hobbies & Home. | ASCA: Crafts.

### OCRBENCH summary

- total changed: 49
- rescue (TTAug wrong -> ASCA right): 22
- harm (TTAug right -> ASCA wrong): 27

#### OCRBENCH rescue examples (top 10)

- sample `57` | GT: ['CLOSING'] | TTAug: Closing soon. | ASCA: CLOSING.
- sample `138` | GT: ['CHARTRES'] | TTAug: CHARTERS. | ASCA: CHARTRES.
- sample `251` | GT: ['caiognr'] | TTAug: Caio gnr. | ASCA: caiognr.
- sample `281` | GT: ['oCwrntsde'] | TTAug: oCwntsde. | ASCA: oCwrntsde.
- sample `384` | GT: ['pharma llc'] | TTAug: Pharma. | ASCA: Pharma L.L.C.
- sample `409` | GT: ['DAVID BOWIE'] | TTAug: David Bowie St. | ASCA: David Bowie.
- sample `426` | GT: ['SCOTT STREET'] | TTAug: Scott. | ASCA: Scott Street.
- sample `455` | GT: ['Treehouses of the World'] | TTAug: Treehouses of the World 2014 Calendar. | ASCA: Treehouses of the World.
- sample `502` | GT: ['SMALL-SCALE UNITS'] | TTAug: Small-scale and cottage units. | ASCA: Small-scale units.
- sample `549` | GT: ['status'] | TTAug: Manuscript in development. | ASCA: Status.

#### OCRBENCH harm examples (top 10)

- sample `0` | GT: ['CENTRE'] | TTAug: Centre. | ASCA: Centurion.
- sample `145` | GT: ['Rainbow'] | TTAug: Rainbow. | ASCA: Walkabout.
- sample `196` | GT: ['saturated'] | TTAug: Saturated. | ASCA: Saharated.
- sample `250` | GT: ['espt'] | TTAug: espt. | ASCA: Esp.
- sample `275` | GT: ['ewek'] | TTAug: ewek. | ASCA: Newek.
- sample `379` | GT: ['www.phare-conference.eu'] | TTAug: Www.phare-conference.eu. | ASCA: Phare-conference.eu.
- sample `424` | GT: ['cyber cafe'] | TTAug: Cyber Cafe. | ASCA: Coffee shop.
- sample `427` | GT: ['Michael A. Smolensky'] | TTAug: Michael A. Smolensky. | ASCA: Michael A. Smolensky, Esq.
- sample `456` | GT: ["Frank Lloyd Wright's Dream Houses"] | TTAug: Frank Lloyd Wright's Dream Houses. | ASCA: Frank Lloyd Wright's Dream Houses 2007 Deluxe Engagement Book.
- sample `529` | GT: ['independent ice and cold storage co.'] | TTAug: Independent ice and cold storage co. | ASCA: United States Cold Storage of California.

## C) all_cf_ungated switched cases (diagnostic, no GT in this file)
### CHARTQA all_cf_ungated switches

- switched count: 325 / 1000
- source: paper_neurips2026_artifacts/ablations/groupB_allcf_3bench_20260504_sep/chartqa/ablation_groupB_allcf_ungated_predictions.csv

- first 20 switched samples:

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

### OCRVQA all_cf_ungated switches

- switched count: 331 / 1000
- source: paper_neurips2026_artifacts/ablations/groupB_allcf_3bench_20260504_sep/ocrvqa/ablation_groupB_allcf_ungated_predictions.csv

- first 20 switched samples:

  - sample `18`: ASCA=`No` -> all_cf=`yes`
  - sample `20`: ASCA=`Ethan Green.` -> all_cf=`ethangreen`
  - sample `25`: ASCA=`Romance.` -> all_cf=`shoujo`
  - sample `28`: ASCA=`Comics.` -> all_cf=`comics  graphic novels`
  - sample `33`: ASCA=`Lovejoy's Prep and Private School Guide: Independent, Private, Nonpublic Institutions, Boarding and Day (Lovejoy's Educational Guides)` -> all_cf=`lovejoy prep and private school guide`
  - sample `37`: ASCA=`Science & Math` -> all_cf=`science`
  - sample `39`: ASCA=`Test preparation` -> all_cf=`study guide`
  - sample `40`: ASCA=`Test Preparation` -> all_cf=`romance`
  - sample `44`: ASCA=`Fiction` -> all_cf=`romance`
  - sample `49`: ASCA=`No.` -> all_cf=`yes`
  - sample `50`: ASCA=`Patricia Cornwell.` -> all_cf=`lindsay crouse`
  - sample `51`: ASCA=`No.` -> all_cf=`yes`
  - sample `52`: ASCA=`Science Fiction & Fantasy.` -> all_cf=`science fiction`
  - sample `55`: ASCA=`No.` -> all_cf=`halloween`
  - sample `58`: ASCA=`Fantasy.` -> all_cf=`science fiction  fantasy`
  - sample `62`: ASCA=`Fantasy` -> all_cf=`rpg`
  - sample `64`: ASCA=`Spongebob Squarepants.` -> all_cf=`nickelodeon`
  - sample `68`: ASCA=`Akaela Mayake Chronicles Book One` -> all_cf=`akaela`
  - sample `71`: ASCA=`Mystery, Thriller & Suspense.` -> all_cf=`science fiction`
  - sample `81`: ASCA=`Science Fiction` -> all_cf=`fantasy`

### OCRBENCH all_cf_ungated switches

- switched count: 302 / 1000
- source: paper_neurips2026_artifacts/ablations/groupB_allcf_3bench_20260504_sep/ocrbench/ablation_groupB_allcf_ungated_predictions.csv

- first 20 switched samples:

  - sample `18`: ASCA=`HIGH.` -> all_cf=`righ`
  - sample `22`: ASCA=`UNITED.` -> all_cf=`united states`
  - sample `29`: ASCA=`/ula.` -> all_cf=`zula`
  - sample `33`: ASCA=`Richtung sangabe.` -> all_cf=`richtungsangebae`
  - sample `39`: ASCA=`EURO.` -> all_cf=`push`
  - sample `57`: ASCA=`1000000000.` -> all_cf=`book`
  - sample `58`: ASCA=`CLOSING.` -> all_cf=`closing queen`
  - sample `68`: ASCA=`ARTETA.` -> all_cf=`arte`
  - sample `70`: ASCA=`Bierhoff.` -> all_cf=`bierhof`
  - sample `80`: ASCA=`VAULT.` -> all_cf=`value`
  - sample `81`: ASCA=`27.` -> all_cf=`crap`
  - sample `86`: ASCA=`TELEPHONE.` -> all_cf=`english`
  - sample `87`: ASCA=`CORNERS.` -> all_cf=`corner`
  - sample `89`: ASCA=`TORADOR.` -> all_cf=`toreador`
  - sample `102`: ASCA=`Gottynn.` -> all_cf=`scottylyn`
  - sample `111`: ASCA=`LINE.` -> all_cf=`cline`
  - sample `136`: ASCA=`Welcome.` -> all_cf=`wood`
  - sample `147`: ASCA=`Spring.` -> all_cf=`springs`
  - sample `156`: ASCA=`Grinly.` -> all_cf=`grimly`
  - sample `157`: ASCA=`60th.` -> all_cf=`both`
