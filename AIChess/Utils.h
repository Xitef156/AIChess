#pragma once

std::unordered_map<Team, int> checkmateStreak;

bool isWithinBounds(const Tensor<Cell>& board, const std::vector<size_t>& position)
{
    return position[0] < board.shape[0] && position[1] < board.shape[1];
}

bool isLinearMoveAndClear(const Tensor<Cell>& board,
    const std::vector<size_t>& from,
    const std::vector<size_t>& to)
{
    // Vérifie si la destination est dans les limites
    if (!isWithinBounds(board, to))
    {
        return false; // Destination hors des limites
    }

    // Vérifie si le mouvement est horizontal ou vertical
    if (from[0] != to[0] && from[1] != to[1])
    {
        return false; // Pas un mouvement linéaire
    }

    // Détermine la direction du mouvement
    int xStep = (to[0] > from[0]) ? 1 : ((to[0] < from[0]) ? -1 : 0);
    int yStep = (to[1] > from[1]) ? 1 : ((to[1] < from[1]) ? -1 : 0);

    // Parcourt toutes les cases intermédiaires
    size_t x = from[0], y = from[1];
    while (true)
    {
        x += xStep;
        y += yStep;

        // Vérifie si la position intermédiaire est dans les limites
        if (!isWithinBounds(board, { x, y }))
        {
            return false; // Hors des limites
        }

        if (x == to[0] && y == to[1])
            break; // Arrivé à destination

        if (board.at({ x, y }).pieceType != 0)
        {
            return false; // Une pièce bloque le chemin
        }
    }

    return true; // Chemin libre
}

bool isDiagonalMoveAndClear(const Tensor<Cell>& board,
    const std::vector<size_t>& from,
    const std::vector<size_t>& to)
{
    // Vérifie si la destination est dans les limites
    if (!isWithinBounds(board, to))
    {
        return false; // Destination hors des limites
    }

    // Vérifie si le mouvement est diagonal
    int dx = std::abs((int)to[0] - (int)from[0]);
    int dy = std::abs((int)to[1] - (int)from[1]);
    if (dx != dy)
    {
        return false; // Pas un mouvement diagonal
    }

    // Détermine la direction du mouvement
    int xStep = (to[0] > from[0]) ? 1 : -1;
    int yStep = (to[1] > from[1]) ? 1 : -1;

    // Parcourt toutes les cases intermédiaires
    size_t x = from[0], y = from[1];
    while (true)
    {
        x += xStep;
        y += yStep;

        // Vérifie si la position intermédiaire est dans les limites
        if (!isWithinBounds(board, { x, y }))
        {
            return false; // Hors des limites
        }

        if (x == to[0] && y == to[1])
            break; // Arrivé à destination

        if (board.at({ x, y }).pieceType != 0)
        {
            return false; // Une pièce bloque le chemin
        }
    }

    return true; // Chemin libre
}

void definePieceMovements(GameManager& game)
{
    // Déplacement des pions
    game.addGlobalPiece(1, [](const Tensor<Cell>& board, const std::vector<size_t>& from, const std::vector<size_t>& to, Team team) -> std::pair<std::vector<size_t>, bool> {
        // Vérifie si la destination est dans les limites
        if (!isWithinBounds(board, to))
        {
            return { {}, false }; // Destination hors des limites
        }

        int direction = (team == 1 ? 1 : -1);

        // Déplacement d'une case vers l'avant
        if (to[0] == from[0] + direction && to[1] == from[1] && board.at(to).pieceType == 0)
        {
            return { to, true };
        }

        // Double pas initial
        if (to[0] == from[0] + 2 * direction && to[1] == from[1] &&
            board.at(to).pieceType == 0 && board.at({ from[0] + direction, from[1] }).pieceType == 0 &&
            ((team == 1 && from[0] == 1) || (team == 2 && from[0] == 6)))
        {
            return { to, true };
        }

        // Capture diagonale
        if (to[0] == from[0] + direction && std::abs((int)(to[1]) - (int)(from[1])) == 1 &&
            board.at(to).pieceType != 0 && board.at(to).team != team)
        {
            return { to, true };
        }

        return { {}, false }; // Déplacement invalide
        });

    // Déplacement des cavaliers
    game.addGlobalPiece(2, [](const Tensor<Cell>& board, const std::vector<size_t>& from, const std::vector<size_t>& to, Team team) -> std::pair<std::vector<size_t>, bool> {
        // Vérifie si la destination est dans les limites
        if (!isWithinBounds(board, to))
        {
            return { {}, false }; // Destination hors des limites
        }

        int dx = std::abs((int)(to[0]) - (int)(from[0]));
        int dy = std::abs((int)(to[1]) - (int)(from[1]));
        if ((dx == 2 && dy == 1) || (dx == 1 && dy == 2))
        {
            if (board.at(to).pieceType == 0 || board.at(to).team != team)
            {
                return { to, true };
            }
        }
        return { {}, false };
        });

    // Déplacement des fous
    game.addGlobalPiece(3, [](const Tensor<Cell>& board, const std::vector<size_t>& from, const std::vector<size_t>& to, Team team) -> std::pair<std::vector<size_t>, bool> {
        if (!isWithinBounds(board, to))
        {
            return { {}, false }; // Destination hors des limites
        }

        if (!isDiagonalMoveAndClear(board, from, to))
        {
            return { {}, false }; // Le fou ne peut se déplacer qu'en diagonale avec un chemin libre
        }

        const Cell& targetCell = board.at(to);
        if (targetCell.pieceType == 0 || targetCell.team != team)
        {
            return { to, true };
        }

        return { {}, false };
        });

    // Déplacement des tours
    game.addGlobalPiece(4, [](const Tensor<Cell>& board, const std::vector<size_t>& from, const std::vector<size_t>& to, Team team) -> std::pair<std::vector<size_t>, bool> {
        if (!isWithinBounds(board, to))
        {
            return { {}, false }; // Destination hors des limites
        }

        if (!isLinearMoveAndClear(board, from, to))
        {
            return { {}, false }; // La tour ne peut se déplacer qu'en ligne droite avec un chemin libre
        }

        const Cell& targetCell = board.at(to);
        if (targetCell.pieceType == 0 || targetCell.team != team)
        {
            return { to, true };
        }

        return { {}, false };
        });

    // Déplacement de la dame
    game.addGlobalPiece(5, [](const Tensor<Cell>& board, const std::vector<size_t>& from, const std::vector<size_t>& to, Team team) -> std::pair<std::vector<size_t>, bool> {
        if (!isWithinBounds(board, to))
        {
            return { {}, false }; // Destination hors des limites
        }

        if (!isLinearMoveAndClear(board, from, to) &&
            !isDiagonalMoveAndClear(board, from, to))
        {
            return { {}, false }; // La dame doit se déplacer en ligne droite ou en diagonale avec un chemin libre
        }

        const Cell& targetCell = board.at(to);
        if (targetCell.pieceType == 0 || targetCell.team != team)
        {
            return { to, true };
        }

        return { {}, false };
        });

    // Déplacement du roi
    game.addGlobalPiece(6, [](const Tensor<Cell>& board, const std::vector<size_t>& from, const std::vector<size_t>& to, Team team) -> std::pair<std::vector<size_t>, bool> {
        if (!isWithinBounds(board, to))
        {
            return { {}, false }; // Destination hors des limites
        }

        int dx = std::abs((int)(to[0]) - (int)(from[0]));
        int dy = std::abs((int)(to[1]) - (int)(from[1]));

        // Le roi peut se déplacer d'une case dans toutes les directions
        if ((dx <= 1 && dy <= 1))
        {
            const Cell& targetCell = board.at(to);
            if (targetCell.pieceType == 0 || targetCell.team != team)
            {
                return { to, true };
            }
        }

        return { {}, false };
        });
}

void placePieces(GameManager& game)
{
    const size_t boardRows = game.board.shape[0];
    const size_t boardCols = game.board.shape[1];

    // Vérifie que le plateau est au moins 8x8 pour placer les pièces correctement
    if (boardRows < 8 || boardCols < 8)
    {
        throw std::runtime_error("Le plateau doit être au moins de taille 8x8 pour placer les pièces.");
    }

    // Placement des pions
    for (size_t i = 0; i < 8; ++i)
    {
        game.addPieceToTeam(1, 1, { 1, i }, "p"); // Pions blancs (équipe 1)
        game.addPieceToTeam(2, 1, { 6, i }, "P"); // Pions noirs (équipe 2)
    }

    // Placement des tours
    game.addPieceToTeam(1, 4, { 0, 0 }, "r");
    game.addPieceToTeam(1, 4, { 0, 7 }, "r");
    game.addPieceToTeam(2, 4, { 7, 0 }, "R");
    game.addPieceToTeam(2, 4, { 7, 7 }, "R");

    // Placement des cavaliers
    game.addPieceToTeam(1, 2, { 0, 1 }, "n");
    game.addPieceToTeam(1, 2, { 0, 6 }, "n");
    game.addPieceToTeam(2, 2, { 7, 1 }, "N");
    game.addPieceToTeam(2, 2, { 7, 6 }, "N");

    // Placement des fous
    game.addPieceToTeam(1, 3, { 0, 2 }, "b");
    game.addPieceToTeam(1, 3, { 0, 5 }, "b");
    game.addPieceToTeam(2, 3, { 7, 2 }, "B");
    game.addPieceToTeam(2, 3, { 7, 5 }, "B");

    // Placement des dames
    game.addPieceToTeam(1, 5, { 0, 3 }, "q");
    game.addPieceToTeam(2, 5, { 7, 3 }, "Q");

    // Placement des rois
    game.addPieceToTeam(1, 6, { 0, 4 }, "k");
    game.addPieceToTeam(2, 6, { 7, 4 }, "K");
}

bool isPositionUnderAttack(const Tensor<Cell>& board,
    const std::vector<size_t>& position,
    Team team,
    const std::unordered_map<Team, std::vector<PieceInstance>>& teams)
{
    for (const auto& [enemyTeamID, enemyPieces] : teams)
    {
        // Parcourir toutes les pièces ennemies
        for (const auto& enemyPiece : enemyPieces)
        {
            const auto& from = enemyPiece.position;

            // Vérifier si cette pièce ennemie peut atteindre la position donnée
            auto moveResult = enemyPiece.piece->move(board, from, position, enemyTeamID);

            // Si le mouvement est valide et atteint la position cible
            if (moveResult.second && moveResult.first == position)
            {
                return true; // La position est attaquée
            }
        }
    }
    return false; // Aucune pièce ennemie n'attaque cette position
}

// Ajoute en haut du fichier (avant toute fonction) :
enum class GameResult
{
    None = -1,
    Draw = 0,
    Team1 = 1,
    Team2 = 2
};

// Fonction utilitaire pour détecter le matériel insuffisant
bool isInsufficientMaterial(const std::unordered_map<Team, std::vector<PieceInstance>>& teams)
{
    int nonKingPieces = 0;
    for (const auto& [teamID, pieces] : teams)
    {
        for (const auto& piece : pieces)
        {
            if (piece.piece->type != 6) // 6 = roi
                nonKingPieces++;
        }
    }
    // Roi contre roi
    if (nonKingPieces == 0)
        return true;
    // Roi contre roi + fou ou roi + cavalier
    if (nonKingPieces == 1)
    {
        for (const auto& [teamID, pieces] : teams)
        {
            for (const auto& piece : pieces)
            {
                if (piece.piece->type == 3 || piece.piece->type == 2) // fou ou cavalier
                    return true;
            }
        }
    }
    return false;
}

void defineVictoryCondition(GameManager& game)
{
    game.victoryFunction = [&]() -> std::tuple<std::vector<std::string>, bool, int> {
        // Matériel insuffisant
        if (isInsufficientMaterial(game.teams))
        {
            return { {"Match nul : matériel insuffisant pour mater."}, true, 0 };
        }

        for (const auto& [teamID, pieces] : game.teams)
        {
            bool kingAlive = false;
            std::vector<size_t> kingPosition;

            // Trouver le roi et sa position
            for (const auto& pieceInstance : pieces)
            {
                if (pieceInstance.piece->type == 6)
                { // Roi
                    kingAlive = true;
                    kingPosition = pieceInstance.position;
                    break;
                }
            }

            // Si le roi est capturé, toutes les autres équipes encore en jeu gagnent
            if (!kingAlive || kingPosition.empty())
            {
                // Cherche la première équipe encore en vie (différente de teamID)
                for (const auto& [otherTeamID, otherPieces] : game.teams)
                {
                    if (otherTeamID != teamID)
                    {
                        return { {"L'équipe " + std::to_string(otherTeamID) + " gagne ! Le roi de l'équipe " + std::to_string(teamID) + " a été capturé."}, true, static_cast<int>(otherTeamID) };
                    }
                }
                // Si aucune autre équipe, match nul
                return { {"Match nul !"}, true, 0 };
            }

            // Vérifier si le roi est en échec
            bool kingInCheck = isPositionUnderAttack(game.board, kingPosition, teamID, game.teams);

            // Vérifier s'il existe un coup légal pour sortir de l'échec (ou jouer tout court)
            bool hasLegalMove = false;
            for (const auto& pieceInstance : pieces)
            {
                const auto& from = pieceInstance.position;
                for (size_t x = 0; x < game.board.shape[0]; ++x)
                {
                    for (size_t y = 0; y < game.board.shape[1]; ++y)
                    {
                        std::vector<size_t> to = { x, y };
                        auto moveResult = pieceInstance.piece->move(game.board, from, to, teamID);
                        if (!moveResult.second || moveResult.first.empty())
                            continue;

                        // Simuler le mouvement
                        Cell originalToCell = game.board.at(to);
                        Cell originalFromCell = game.board.at(from);
                        game.board.at(to).pieceType = pieceInstance.piece->type;
                        game.board.at(to).team = teamID;
                        game.board.at(from).pieceType = 0;
                        game.board.at(from).team = 0;

                        // Vérifier si le roi reste hors d'échec
                        bool stillInCheck = isPositionUnderAttack(game.board, kingPosition, teamID, game.teams);

                        // Annuler la simulation
                        game.board.at(to) = originalToCell;
                        game.board.at(from) = originalFromCell;

                        if (!stillInCheck)
                        {
                            hasLegalMove = true;
                            break;
                        }
                    }
                    if (hasLegalMove)
                        break;
                }
                if (hasLegalMove)
                    break;
            }

            // Échec et mat
            if (kingInCheck && !hasLegalMove)
            {
                // Cherche la première équipe encore en vie (différente de teamID)
                for (const auto& [otherTeamID, otherPieces] : game.teams)
                {
                    if (otherTeamID != teamID)
                    {
                        return { {"L'équipe " + std::to_string(otherTeamID) + " gagne ! Échec et mat contre l'équipe " + std::to_string(teamID) + "."}, true, static_cast<int>(otherTeamID) };
                    }
                }
                // Si aucune autre équipe, match nul
                return { {"Match nul !"}, true, 0 };
            }

            // Pat
            if (!kingInCheck && !hasLegalMove)
            {
                return { {"Pat ! Match nul : l'équipe " + std::to_string(teamID) + " n'a plus de coup légal."}, true, 0 };
            }
        }
        return { {}, false, -1 };
        };
}

void chessEventFunction(
    GameManager* game,
    Tensor<Cell>& board,
    std::unordered_map<Team, std::vector<PieceInstance>>& teams,
    const std::vector<Piece>& globalPieces,
    Team currentTeam,
    size_t turnNumber)
{
    for (auto& [teamID, pieces] : teams)
    {
        for (auto& piece : pieces)
        {
            if (piece.piece->type == 1)
            {
                size_t lastRow = (teamID == 1 ? board.shape[0] - 1 : 0);

                if (piece.position[0] == lastRow)
                {
                    Pieces newPieceType = 5; // Dame par défaut
                    if (game && game->promotionCallback)
                    {
                        newPieceType = game->promotionCallback(piece);
                    }
                    // Trouver le type de pièce correspondant
                    piece.piece =
                        &*std::find_if(globalPieces.begin(), globalPieces.end(),
                            [&](const Piece& p) { return p.type == newPieceType; });
                    // Mettre à jour le caractère d'affichage
                    char displayChar = 'q';
                    switch (newPieceType)
                    {
                    case 5: displayChar = 'q'; break;
                    case 4: displayChar = 'r'; break;
                    case 3: displayChar = 'b'; break;
                    case 2: displayChar = 'n'; break;
                    }
                    piece.displayChar = (teamID == 1 ? tolower(displayChar) : toupper(displayChar));
                }
            }
        }
    }
}
