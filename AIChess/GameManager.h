#pragma once

class GameManager
{
public:
    Tensor<Cell> board;
    std::unordered_map<Team, std::vector<PieceInstance>> teams;
    std::vector<Piece> globalPieces;
    Team currentTeam;
    size_t currentTurn;

    // Nouvelle signature : messages, partie finie, gagnant (0 = nul, >0 = équipe gagnante, -1 = partie en cours)
    std::function<std::tuple<std::vector<std::string>, bool, int>()> victoryFunction;

    // Correction de la déclaration (suppression de la parenthèse en trop)
    std::function<void(Tensor<Cell>&,
        std::unordered_map<Team, std::vector<PieceInstance>>&,
        const std::vector<Piece>&, Team, size_t)>
        eventFunction;

    // Nouveau : callback pour la promotion
    std::function<Pieces(const PieceInstance&)> promotionCallback;

    GameManager(const std::vector<size_t>& boardShape)
        : board(boardShape), currentTeam(1), currentTurn(0), victoryFunction(nullptr) {
    }

    void callEventFunction()
    {
        if (eventFunction)
            eventFunction(board, teams, globalPieces, currentTeam, currentTurn);
    }

    void addGlobalPiece(Pieces type, moveFunction move)
    {
        globalPieces.emplace_back(type, move);
    }

    void addPieceToTeam(Team team, Pieces type,
        const std::vector<size_t>& position,
        const std::string& displayChar)
    {
        for (const auto& piece : globalPieces)
        {
            if (piece.type == type)
            {
                teams[team].emplace_back(&piece, team, position, displayChar);
                board.at(position).pieceType = type;
                board.at(position).team = team;
                return;
            }
        }
        throw std::runtime_error("Type de pièce non trouvé !");
    }

    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> getMoves()
    {
        std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> coupsPossibles;

        for (auto& pieceInstance : teams[currentTeam])
        {
            std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> mouvements = pieceInstance.getPossibleMoves(board);

            for (auto move : mouvements)
            {
                coupsPossibles.push_back(move);
            }
        }

        return coupsPossibles;
    }

    bool executeMove(const std::vector<size_t>& from,
        const std::vector<size_t>& to)
    {
        // std::cout << "\nTentative de déplacement de (" << from[0] << ", " << from[1] << ") à (" << to[0] << ", " << to[1] << ").\n";

        for (auto& pieceInstance : teams[currentTeam])
        {
            if (pieceInstance.position == from)
            {
                auto moveResult = pieceInstance.piece->move(board,
                    pieceInstance.position,
                    to,
                    currentTeam);
                const auto& newPosition = moveResult.first;
                bool countsAsTurn = moveResult.second;

                if (!newPosition.empty() && newPosition == to)
                {
                    Cell& targetCell = board.at(newPosition);
                    if (targetCell.pieceType != 0 && targetCell.team != currentTeam)
                    {
                        auto& enemyPieces = teams[targetCell.team];
                        enemyPieces.erase(std::remove_if(enemyPieces.begin(), enemyPieces.end(),
                            [&](const PieceInstance& p) {
                                return p.position == newPosition;
                            }),
                            enemyPieces.end());
                    }

                    board.at(pieceInstance.position).pieceType = 0;
                    board.at(pieceInstance.position).team = 0;

                    pieceInstance.position = newPosition;

                    board.at(newPosition).pieceType = pieceInstance.piece->type;
                    board.at(newPosition).team = currentTeam;

                    if (countsAsTurn)
                        nextTurn();

                    return true;
                }
            }
        }

        std::cout << "Déplacement invalide ! Réessayez.\n";
        return false;
    }

    void nextTurn()
    {
        currentTurn++;
        currentTeam = (currentTeam % teams.size()) + 1;
        callEventFunction();
    }

    void displayBoard() const
    {
        size_t maxRowIndex = board.shape[0] - 1;
        size_t maxColIndex = board.shape[1] - 1;

        size_t rowWidth = std::to_string(maxRowIndex).size();
        size_t colWidth = std::to_string(maxColIndex).size();

        std::cout << std::string(rowWidth + 1, ' ');
        for (size_t j = 0; j <= maxColIndex; ++j)
        {
            std::cout << std::setw(colWidth) << j << " ";
        }
        std::cout << "\n";

        for (int i = maxRowIndex; i >= 0; --i)
        {
            std::cout << std::setw(rowWidth) << i << " ";

            for (size_t j = 0; j <= maxColIndex; ++j)
            {
                const Cell& cell = board.at({ (size_t)i, j });
                if (cell.pieceType != 0)
                {
                    bool found = false;
                    for (const auto& [team, pieces] : teams)
                    {
                        for (const auto& pieceInstance : pieces)
                        {
                            if (pieceInstance.position == std::vector<size_t>{(size_t)i, j})
                            {
                                found = true;
                                std::cout << pieceInstance.displayChar << " ";
                                break;
                            }
                        }
                        if (found)
                            break;
                    }
                }
                else
                {
                    std::cout << ". ";
                }
            }
            std::cout << "\n";
        }
    }
};
