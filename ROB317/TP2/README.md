# ROB317 TP2 :  Découpage et indexation des vidéos

Auteurs : Gabriel Henrique Riqueti et Victor Kenichi Nascimento Kobayashi

Orientateurs : Antoine Manzanera et David Filliat

## Objectif

L'objectif de ce TP est de se familiariser avec le traitement de vidéos sur OpenCV, en particulier
l'estimation du champ de vitesses apparentes (flot optique), et de travailler la problématique de
l'indexation automatique de vidéos, dont le but est de sectionner la vidéos en plans, en attachant à
chaque plan une description textuelle, ainsi qu'une image représentative du plan.

## Organisation

### Dossiers

Cet travail est composé par les dossiers décrits suivants :

- Images : stocke des images correspondant aux cadres d'un vidéo ;
- src : contient des fichiers qui ont pour but réaliser les tâches demandées par les questions du TP2 ;
- Validation : contient des fichier python qui ont pour objectif générer les archives CSV qui tiennent l'identification des raccords et des mouvements de caméra de chaque vidéo ;
- Vidéos : posséde les vidéos de test.

### Fichiers Python

#### Sources (src)

- Hist2Duv.py : Q1;
- HistGray.py : Q1.
- Hist2DFlotOptique.py : Q3;
- DecoupagePlans.py : Q4;


#### Validation

- SaveImgfromVideos.py : sauve les cadres de chaque vidéo dans le dossier Images.
- CreateCSVMontage*.py : écrits d'après l'analyse manuellement des cadres de chaque vidéos, il iditifie ses raccords et ses mouvements;
