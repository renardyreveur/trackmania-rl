/*
 * author: Renardy
 * v1.0
 */

namespace Data
{
    // Control print and data collecting condition
    bool playing = true;


    // Class that holds data structure for RL Reward calculation
    class FrameData
    {
        string checkpoint;
        float front_speed;
        float side_speed;
        float distance;
        int duration;
        bool race_finished;
        vec3 position;

        string EncodeJSON()
        {
            return "{\"checkpoint\":" + "\"" + checkpoint + "\"" +"," +
                    "\"front_speed\":" + front_speed + "," +
                    "\"side_speed\":" + side_speed + "," +
                    "\"position\":" + "[" + position.x + "," +  position.y + "," + position.z + "]," +
                    "\"distance\":" + distance + "," +
                    "\"duration\":" + duration + "," +
                    "\"race_finished\":" + race_finished +
                    "}\n";
        }

    }

    // Method that will run every frame collecting agent state
    string Collect()
    {
        // Get Game context
        auto app = GetApp();
        CSmArenaClient@ playground = cast<CSmArenaClient>(app.CurrentPlayground);

        // Using the Vehicle State Plugin Dependency
        auto visState = VehicleState::ViewingPlayerState();

        // Check if we are playing the game (not in menu etc.)
        if (playground is null || visState is null){
            return "";
        }

        // Game artifacts
        auto gameTerminal = playground.GameTerminals[0];
        auto arena = playground.Arena;
        // auto player = cast<CSmScriptPlayer>(cast<CSmPlayer>(gameTerminal.ControlledPlayer).ScriptAPI);
        auto player = cast<CSmScriptPlayer>(cast<CSmPlayer>(arena.Players[0]).ScriptAPI);
        MwFastBuffer<CGameScriptMapLandmark@> landmarks = arena.MapLandmarks;

        // Get current in-game time (for race record and duration)
        auto now = cast<CSmArenaRulesMode>(app.PlaygroundScript).Now; // To the closest 10 milliseconds


        // Collect Data while playing
        FrameData fd;
        if (gameTerminal.UISequence_Current == CGamePlaygroundUIConfig::EUISequence::Playing){
            playing = true;
            int c_landmark_idx = arena.Players[0].CurrentLaunchedRespawnLandmarkIndex;
            auto c_landmark = landmarks[c_landmark_idx];
            string landmark_order;
            if (c_landmark.Tag == "Spawn"){
                landmark_order = "#-1";
            }
            else{
                landmark_order = c_landmark.Waypoint.IdName;
            }
            fd.checkpoint = landmark_order;
            fd.front_speed = visState.FrontSpeed * 3.6f;
            fd.side_speed = VehicleState::GetSideSpeed(visState) * 3.6f;
            fd.position = player.Position;
            fd.distance = player.Distance;
            fd.duration = player.CurrentRaceTime;
            fd.race_finished = false;
        }

        // Race Finished
        if (gameTerminal.UISequence_Current == CGamePlaygroundUIConfig::EUISequence::Finish && playing){
            playing = false;
            fd.race_finished = true;
            fd.duration = now - player.StartTime;
            print("Finished the race with a time of " + (now - player.StartTime)/1000.f + " seconds!");
            return fd.EncodeJSON();
        }

        if (playing) {
            return fd.EncodeJSON();
        }else{
            return "";
        }

    }

}