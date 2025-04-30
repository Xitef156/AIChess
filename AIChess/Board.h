#pragma once

std::vector<std::vector<size_t>> getAllCombinations(const std::vector<size_t>& sizes)
{
    std::vector<std::vector<size_t>> results;

    // Initialiser un vecteur pour suivre les indices courants
    std::vector<size_t> current(sizes.size(), 0);

    // Tant que toutes les combinaisons n'ont pas été générées
    while (true)
    {
        // Ajouter la combinaison actuelle aux résultats
        results.push_back(current);

        // Incrémenter les indices dans l'ordre
        int i = sizes.size() - 1;
        while (i >= 0 && ++current[i] == sizes[i])
        {
            current[i] = 0; // Réinitialiser cet indice
            --i;            // Passer au niveau précédent
        }

        if (i < 0)
        {
            break;
        }
    }

    return results;
}

typedef unsigned int Pieces;
typedef unsigned int Team;

struct Cell
{
    Pieces pieceType;
    Team team;

    Cell(Pieces type = 0, Team t = 0) : pieceType(type), team(t) {}
};

typedef std::function<std::pair<std::vector<size_t>, bool>(
    const Tensor<Cell>&, const std::vector<size_t>&,
    const std::vector<size_t>&, Team)>
    moveFunction;

struct Piece
{
    Pieces type;
    moveFunction move;

    Piece(Pieces type, moveFunction move)
        : type(type), move(move) {
    }
};

struct PieceInstance
{
    const Piece* piece;
    Team team;
    std::vector<size_t> position;
    std::string displayChar;

    PieceInstance(const Piece* piece, Team team,
        const std::vector<size_t>& position,
        const std::string& displayChar)
        : piece(piece), team(team), position(position), displayChar(displayChar) {
    }

    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> getPossibleMoves(Tensor<Cell>& board)
    {
        std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> coups;

        std::vector<std::vector<size_t>> comb = getAllCombinations(board.shape);
        for (const auto& combi : comb)
        {
            if (combi[0] >= board.shape[0] || combi[1] >= board.shape[1])
            {
                continue; // Ignorer les mouvements hors limites
            }
            try
            {
                std::pair<std::vector<size_t>, bool> p = piece->move(board, position, combi, team);
                if (p.second)
                {
                    coups.emplace_back(position, p.first);
                }
            }
            catch (const std::exception& e)
            {
                std::cerr << "Erreur rencontrée : " << e.what() << "\n";
            }
        }
        return coups;
    }
};

// Fonction pour convertir un Cell en double
double cellToDouble(const Cell& cell)
{
    // Encodage bijectif : utiliser une formule unique
    // Exemple : encoder pieceType et team dans un seul double
    return static_cast<double>(cell.pieceType) + static_cast<double>(cell.team) * 100.0;
}

// Fonction pour convertir un double en Cell
Cell doubleToCell(double value)
{
    // Décodage bijectif : récupérer pieceType et team
    unsigned int team = static_cast<unsigned int>(std::floor(value / 100.0));
    unsigned int pieceType = static_cast<unsigned int>(std::round(value - team * 100.0));
    return Cell(pieceType, team);
}