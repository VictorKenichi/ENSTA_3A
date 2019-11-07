/*////////////////////////////////////////////////////////////////////////////////
|
| Fichier :                parse_param.cpp
| Auteur :                 RIQUETI Gabriel Henrique
| Date :                   07/11/2019
| Commentaires :           Code bas pour passer des paramètres depuis le teminal
|                          en C++
| Commande :               g++ parse_param.cpp -o ./parse_param && ./parse_param
| Historique de Révision :
|
*/////////////////////////////////////////////////////////////////////////////////

#include <iostream>

int main(int argc, char *argv[])
{
    std::cout << "argc : " << argc << std::endl;

    for (int i=0;i<argc;i++)
    {
        std::cout << "argv[" << i <<"] : " << argv[i] << std::endl;
    }

    return 0;
}
