# Data Documentation


## Dataset Description

Here, you'll find a summary of each data set for this compeition, a list of *key* variables to join on, and a description of each variable. The tracking data is provided by the NFL Next Gen Stats team. The Scouting data is provided by [Pro Football Focus](https://www.pff.com/).

This competition allows participants to use supplemental NFL data as long as it is free and publicly available to all participants. Examples of sources that could be used include [nflverse](https://nflverse.nflverse.com/) and [Pro Football Reference](https://www.pro-football-reference.com/). Please note that the `gameId` and `playId` of the Big Data Bowl data merge with the `old_game_id` and `play_id` of nflverse's play-by-play data.

## File descriptions

Game data: The `games.csv` contains the teams playing in each game. The *key* variable is **`gameId`**.

Play data: The `plays.csv` file contains play-level information from each game. The *key* variables are **`gameId`** and **`playId`**.

Player data: The `players.csv` file contains player-level information from players that participated in any of the tracking data files. The *key* variable is **`nflId`**.

PFF Scouting data: The `pffScoutingData.csv` file contains player-level scouting information for each game and play. The *key* variables are **`gameId`**, **`playId`**, and **`nflId`**.

Tracking data: Files `week[week].csv` contain player tracking data from season `[week]`. The *key* variables are **`gameId`**, **`playId`**, and **`nflId`**.

## Game data

- **`gameId`**: Game identifier, unique (numeric)
- `season`: Season of game
- `week`: Week of game
- `gameDate`: Game Date (time, mm/dd/yyyy)
- `gameTimeEastern`: Start time of game (time, HH:MM:SS, EST)
- `homeTeamAbbr`: Home team three-letter code (text)
- `visitorTeamAbbr`: Visiting team three-letter code (text)

## Play data

- **`gameId`**: Game identifier, unique (numeric)
- **`playId`**: Play identifier, not unique across games (numeric)
- `playDescription`: Description of play (text)
- `quarter`: Game quarter (numeric)
- `down`: Down (numeric)
- `yardsToGo`: Distance needed for a first down (numeric)
- `possessionTeam`: Team abbr of team on offense with possession of ball (text)
- `defensiveTeam`: Team abbr of team on defense (text)
- `yardlineSide`: 3-letter team code corresponding to line-of-scrimmage (text)
- `yardlineNumber`: Yard line at line-of-scrimmage (numeric)
- `gameClock`: Time on clock of play (MM:SS)
- `preSnapHomeScore`: Home score prior to the play (numeric)
- `preSnapVisitorScore`: Visiting team score prior to the play (numeric)
- `passResult`: Dropback outcome of the play (`C`: Complete pass, `I`: Incomplete pass, `S`: Quarterback sack, `IN`: Intercepted pass, `R`: Scramble, text)
- `penaltyYards`: yards gained by offense by penalty (numeric)
- `prePenaltyPlayResult`: Net yards gained by the offense, before penalty yardage (numeric)
- `playResult`: Net yards gained by the offense, including penalty yardage (numeric)
- `foulName[i]`: Name of the i-th penalty committed during the play. i ranges between 1 and 3 (text)
- `foulNFLId[i]`: `nflId` of the player who comitted the i-th penalty during the play. i ranges between 1 and 3 (numeric)
- `absoluteYardlineNumber`: Distance from end zone for possession team (numeric)
- `offenseFormation`: Formation used by possession team (text)
- `personnelO`: Personnel used by offensive team (text)
- `defendersInTheBox`: Number of defenders in close proximity to line-of-scrimmage (numeric)
- `personnelD`: Personnel used by defensive team (text)
- `dropbackType`: Dropback categorization of quarterback (text)
- `pff_playAction`: indicator for whether offense executes play action fake on the play. Variable provided by PFF (binary)
- `pff_passCoverage`: Coverage scheme of defense. Variable provided by PFF (text)
  - Possible values:
    - `Cover-0`: A Man to Man coverage across the board with no deep defenders. This coverage is typically accompanied with a blitz
    - `Cover-1`: When a defense plays any form of Man defense across the board with a Defensive Player as a Single High Man concept
    - `Cover-2`: A two deep safety concept where any zone principle is applied
    - `2-Man`: A two deep safety concept where a man principle is applied
    - `Cover-3`: Any 3 Deep, 4 Under concept
    - `Quarters`: A Quarters concept on both halves of the field. In general it will be a 4 Deep, 3 Under concept where the corners are on #1, safeties on #2, and backside safety rotation dependent on formation
    - `Cover-6`: A Quarters Concept on half the field and a 2 Deep concept on the other half
    - `Bracket`: Recorded in the field and up to the 12 yard line in the red zone – when two offensive players have an in and out bracket by two defenders
    - `Goal Line`: Calls where a Goal Line defense is used.
    - `Red Zone`: Calls that are typically specific to the Red Zone and do not occur in the field often
    - `Prevent`: Special end of half or end of game situations where a Prevent defense is utilized
    - `Miscellaneous`: Coverage concepts that we feel do not comfortably fit into any of our coverage categories
- `pff_passCoverageType`: Whether defense's coverage type was man, zone or other. Variable provided by PFF (text)

## Player data

- **`nflId`**: Player identification number, unique across players (numeric)
- `height`: Player height (text)
- `weight`: Player weight (numeric)
- `birthDate`: Date of birth (YYYY-MM-DD)
- `collegeName`: Player college (text)
- `officialPosition`: Official player position (text)
- `displayName`: Player name (text)

## PFF Scouting data

- **`gameId`**: Game identifier, unique (numeric)
- **`playId`**: Play identifier, not unique across games (numeric)
- **`nflId`**: Player identification number, unique across players (numeric)
- `pff_role`: The player's role on this play (text)
  - Possible values:
    - `Coverage`: Defensive player. Player whose initial goal is to play man or zone coverage
    - `Pass`: Offensive player. Player identified as the passer
    - `Pass block`: Offensive player. Anyone fully blocking a defender from the QB, or anyone in a clear pass block stance
    - `Pass route`: Offensive player. Any player not identified as a Pass Blocker or Passer
    - `Pass rush`: Defensive player. Any player whose initial intent is to rush the passer
- `pff_positionLinedUp`: Position that the player was aligned at the snap of the ball on this play (text)
- `pff_hit`: If player is a defensive player, indicator for whether they are credited with recording a hit on this play (binary)
- `pff_hurry`: If player is a defensive player, indicator for whether they are credited with recording a hurry on this play (binary)
- `pff_sack`: If player is a defensive player, indicator for whether they are credited with recording a sack on this play (binary)
- `pff_beatenByDefender`: If player is a blocking offensive player, indicator for whether they are by a defender but was not charged for yielding a hit, hurry or sack (binary)
- `pff_hitAllowed`: If player is a blocking offensive player, indicator for whether they are responsible for a hit on the QB (binary)
- `pff_hurryAllowed`: If player is a blocking offensive player, indicator for whether they are responsible for a hurry on the QB (binary)
- `pff_sackAllowed`: If player is a blocking offensive player, indicator for whether they are responsible for a sack on the QB (binary)
- `pff_nflIdBlockedPlayer`: If player is a blocking offensive player, the `nflId` of the first defender the offensive player blocked (numeric)
- `pff_blockType`: If player is a blocking offensive player, the type of block that the offensive player is executing on the defender (text)
  - Possible values:
    - `BH`: Backfield Help - A block from a player aligned in the backfield on which the blocker merely helps on a block rather than fully engaging his assignment. Usually seen when a blocker is clearing up a block or picking up a defender when he has broken through or been missed by another blocker
    - `CH`: Chip Block - This is only to be used for players who chip a pass rusher when they release for their route
    - `CL`: Second Level – A block made at the second level, this must be at least two yards across the line of scrimmage
    - `NB`: No Block - If a blocker executes no block on a play but simply runs his path or takes his pass set then we will note him with one all blocking line with this block type
    - `PA`: Play Action Pass Protection - A blocker pass protecting inline on a play action pass selling the play action by stepping in to show a run block before converting to pass protect
    - `PP`: Pass Protection - A standard pass protection block from an inline blocker
    - `PR`: Pocket Roll Block - This block type will be used any time the offense is executing a “rolling pocket” by which the entire offensive line moves with the QB’s rollout to stay in front of him but without ever taking a “conventional” pass set. There will be flexibility here to record the PR – Pocket Roll Block type in the same way as PA & RP block types in that individual matchups & responsibilities won’t always be obvious or necessary, so PR block types can be recorded by multiple blockers on an individual defender on the same play
    - `PT`: Post Block - A post block by an offensive player in pass protection to control a defender for another blocker while clearly demonstrating that he is not, at least initially, trying to fully engage with the block
    - `PU`: Backfield Pickup - A pass protection pick-up by a player aligned in the backfield
    - `SR`: Set & Release - A blocker who sets to pass protect a defender before releasing. This block will cover both players releasing from a set to block for a screen as well as “hold ups” by tight ends before they leak into the flat
    - `SW`: Switch Block - A blocker who passes off (or attempts to pass off) a defender. Most often used on stunts but can also be used for pass offs when pass rushers are slanting across the pocket or interior defenders are working to the edge to replace a dropping edge rusher, with an interior offensive lineman passing them out rather than staying with them
    - `UP`: Pull Pass Protection - A blocker pulling in pass protection from an inline alignment to block a defender in pass protection
- `pff_backFieldBlock`: If player is a blocking offensive player, indicator for whether block occured in offensive backfield.

## Tracking data

Files `week[week].csv` contains player tracking data from week `[week]`.

- **`gameId`**: Game identifier, unique (numeric)
- **`playId`**: Play identifier, not unique across games (numeric)
- **`nflId`**: Player identification number, unique across players. When value is NA, row corresponds to ball. (numeric)
- `frameId`: Frame identifier for each play, starting at 1 (numeric)
- `time`: Time stamp of play (time, yyyy-mm-dd, hh:mm:ss)
- `jerseyNumber`: Jersey number of player (numeric)
- `club`: Team abbrevation of corresponding player (text)
- `playDirection`: Direction that the offense is moving (left or right)
- `x`: Player position along the long axis of the field, 0 - 120 yards. See Figure 1 below. (numeric)
- `y`: Player position along the short axis of the field, 0 - 53.3 yards. See Figure 1 below. (numeric)
- `s`: Speed in yards/second (numeric)
- `a`: Acceleration in yards/second^2 (numeric)
- `dis`: Distance traveled from prior time point, in yards (numeric)
- `o`: Player orientation (deg), 0 - 360 degrees (numeric)
- `dir`: Angle of player motion (deg), 0 - 360 degrees (numeric)
- `event`: Tagged play details, including moment of ball snap, pass release, pass catch, tackle, etc (text)

![Figure 1: Field coordinates, direction, and orientation](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3258%2F820e86013d48faacf33b7a32a15e814c%2FIncreasing%20Dir%20and%20O.png?generation=1572285857588233&alt=media)

## Files

12 files

## Size

965.07 MB

## Type

csv

## License

Subject to Competition Rules

### `games.csv` (6.74 kB)

7 of 7 columns

Columns: `gameId`, `season`, `week`, `gameDate`, `gameTimeEastern`, `homeTeamAbbr`, `visitorTeamAbbr`.

| Range                         | Count |
| ----------------------------- | -- |
| 2021090900.00 - 2021092820.00 | 48 |
| 2021092820.00 - 2021094740.00 | 1  |
| 2021098580.00 - 2021100500.00 | 15 |
| 2021100500.00 - 2021102420.00 | 42 |
| 2021102420.00 - 2021104340.00 | 15 |
| 2021108180.00 - 2021110100.00 | 1  |

**2.02b**

| Range             | Count |
| ----------------- | --- |
| 2021.00 - 2021.00 | 122 |

**2021**

| Range       | Count |
| ----------- | -- |
| 1.00 - 1.70 | 16 |
| 1.70 - 2.40 | 16 |
| 2.40 - 3.10 | 16 |
| 3.80 - 4.50 | 16 |
| 4.50 - 5.20 | 16 |
| 5.90 - 6.60 | 14 |
| 6.60 - 7.30 | 13 |
| 7.30 - 8.00 | 15 |

**1**

**8**

| Date/time range           | Count |
| ----------------------- | -- |
| 09/09/2021 - 09/14/2021 | 16 |
| 09/14/2021 - 09/19/2021 | 15 |
| 09/19/2021 - 09/24/2021 | 2  |
| 09/24/2021 - 09/30/2021 | 16 |
| 09/30/2021 - 10/05/2021 | 15 |
| 10/05/2021 - 10/10/2021 | 15 |
| 10/10/2021 - 10/16/2021 | 2  |
| 10/16/2021 - 10/21/2021 | 14 |
| 10/21/2021 - 10/26/2021 | 12 |
| 10/26/2021 - 11/01/2021 | 15 |

**2021-09-08**

**2021-10-31**

| Date/time range           | Count |
| ----------------------- | -- |
| 10/20/2022 - 10/20/2022 | 2  |
| 10/20/2022 - 10/20/2022 | 65 |
| 10/20/2022 - 10/20/2022 | 31 |
| 10/20/2022 - 10/20/2022 | 24 |

**2022-10-20**

**NE4%**

**CLE4%**

**Other (112)92%**

**PHI4%**

**MIA4%**

**Other (112)92%**

#### Exported data preview

The pasted page joined these preview cells without separators; the original text is preserved below.

```text
20210909002021109/09/202120:20:00TBDAL20210912002021109/12/202113:00:00ATLPHI20210912012021109/12/202113:00:00BUFPIT20210912022021109/12/202113:00:00CARNYJ20210912032021109/12/202113:00:00CINMIN20210912042021109/12/202113:00:00DETSF20210912052021109/12/202113:00:00HOUJAX20210912062021109/12/202113:00:00INDSEA20210912072021109/12/202113:00:00TENARI20210912082021109/12/202113:00:00WASLAC20210912092021109/12/202116:25:00KCCLE20210912102021109/12/202116:25:00NEMIA20210912112021109/12/202116:25:00NOGB20210912122021109/12/202116:25:00NYGDEN20210912132021109/12/202120:20:00LACHI20210913002021109/13/202120:15:00LVBAL20210916002021209/16/202120:20:00WASNYG20210919002021209/19/202113:00:00CARNO20210919012021209/19/202113:00:00CHICIN20210919022021209/19/202113:00:00CLEHOU20210919032021209/19/202113:00:00INDLA20210919042021209/19/202113:00:00JAXDEN20210919052021209/19/202113:00:00MIABUF20210919062021209/19/202113:00:00NYJNE20210919072021209/19/202113:00:00PHISF20210919082021209/19/202113:00:00PITLV20210919092021209/19/202116:05:00ARIMIN20210919102021209/19/202116:05:00TBATL20210919112021209/19/202116:25:00LACDAL20210919122021209/19/202116:25:00SEATEN20210919132021209/19/202120:20:00BALKC20210920002021209/20/202120:15:00GBDET20210923002021309/23/202120:20:00HOUCAR20210926002021309/26/202113:00:00BUFWAS20210926012021309/26/202113:00:00CLECHI20210926022021309/26/202113:00:00DETBAL20210926032021309/26/202113:00:00JAXARI20210926042021309/26/202113:00:00KCLAC20210926052021309/26/202113:00:00NENO20210926062021309/26/202113:00:00NYGATL20210926072021309/26/202113:00:00PITCIN20210926082021309/26/202113:00:00TENIND20210926092021309/26/202116:05:00DENNYJ20210926102021309/26/202116:05:00LVMIA20210926112021309/26/202116:25:00LATB20210926122021309/26/202116:25:00MINSEA20210926132021309/26/202120:20:00SFGB20210927002021309/27/202120:15:00DALPHI20210930002021409/30/202120:20:00CINJAX20211003002021410/03/202113:00:00ATLWAS20211003012021410/03/202113:00:00BUFHOU20211003022021410/03/202113:00:00CHIDET20211003032021410/03/202113:00:00DALCAR20211003042021410/03/202113:00:00MIAIND20211003052021410/03/202113:00:00MINCLE20211003062021410/03/202113:00:00NONYG20211003072021410/03/202113:00:00NYJTEN20211003082021410/03/202113:00:00PHIKC20211003092021410/03/202116:05:00LAARI20211003102021410/03/202116:05:00SFSEA20211003112021410/03/202116:25:00DENBAL20211003122021410/03/202116:25:00GBPIT20211003132021410/03/202120:20:00NETB20211004002021410/04/202120:15:00LACLV20211007002021510/07/202120:20:00SEALA20211010002021510/10/202109:30:00ATLNYJ20211010012021510/10/202113:00:00CARPHI20211010022021510/10/202113:00:00CINGB20211010032021510/10/202113:00:00HOUNE20211010042021510/10/202113:00:00JAXTEN20211010052021510/10/202113:00:00MINDET20211010062021510/10/202113:00:00PITDEN20211010072021510/10/202113:00:00TBMIA20211010082021510/10/202113:00:00WASNO20211010092021510/10/202116:05:00LACCLE20211010102021510/10/202116:05:00LVCHI20211010112021510/10/202116:25:00ARISF20211010122021510/10/202116:25:00DALNYG20211010132021510/10/202120:20:00KCBUF20211011002021510/11/202120:15:00BALIND20211014002021610/14/202120:20:00PHITB20211017002021610/17/202109:30:00JAXMIA20211017012021610/17/202113:00:00BALLAC20211017022021610/17/202113:00:00CARMIN20211017032021610/17/202113:00:00CHIGB20211017042021610/17/202113:00:00DETCIN20211017052021610/17/202113:00:00INDHOU20211017062021610/17/202113:00:00NYGLA20211017072021610/17/202113:00:00WASKC20211017082021610/17/202116:05:00CLEARI20211017092021610/17/202116:25:00DENLV20211017102021610/17/202116:25:00NEDAL20211017112021610/17/202120:20:00PITSEA20211018002021610/18/202120:15:00TENBUF20211021002021710/21/202120:20:00CLEDEN20211024002021710/24/202113:00:00BALCIN20211024012021710/24/202113:00:00GBWAS20211024022021710/24/202113:00:00MIAATL20211024032021710/24/202113:00:00NENYJ20211024042021710/24/202113:00:00NYGCAR
```

## Data Explorer

965.07 MB

- games.csv
- pffScoutingData.csv
- players.csv
- plays.csv
- week1.csv
- week2.csv
- week3.csv
- week4.csv
- week5.csv
- week6.csv
- week7.csv
- week8.csv

## Summary

**12 files**

**189 columns**

**Download All**

## Metadata

### License

Subject to Competition Rules
