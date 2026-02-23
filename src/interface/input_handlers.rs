use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::prelude::*;
use bevy::window::{Ime, PrimaryWindow};

use super::*;

pub(super) fn handle_world_tool_button_interaction(
    mut interactions: Query<
        (&Interaction, &WorldToolButton),
        (Changed<Interaction>, With<WorldToolButton>),
    >,
    mut interaction_state: ResMut<WorldInteractionState>,
) {
    for (interaction, button) in &mut interactions {
        if *interaction == Interaction::Pressed {
            select_world_tool(&mut interaction_state, Some(button.tool));
        }
    }
}

pub(super) fn handle_save_load_open_button_interaction(
    mut interactions: Query<
        (&Interaction, &SaveLoadOpenButton),
        (Changed<Interaction>, With<Button>),
    >,
    mut save_load_ui_state: ResMut<SaveLoadUiState>,
) {
    for (interaction, button) in &mut interactions {
        if *interaction != Interaction::Pressed {
            continue;
        }
        open_save_load_dialog(&mut save_load_ui_state, button.mode);
    }
}

pub(super) fn handle_save_load_reset_button_interaction(
    mut interactions: Query<
        &Interaction,
        (
            Changed<Interaction>,
            With<SaveLoadResetButton>,
            With<Button>,
        ),
    >,
    mut reset_writer: MessageWriter<ResetSimulationRequest>,
) {
    for interaction in &mut interactions {
        if *interaction == Interaction::Pressed {
            reset_writer.write(ResetSimulationRequest);
        }
    }
}

pub(super) fn handle_sim_play_pause_button_interaction(
    mut interactions: Query<
        &Interaction,
        (Changed<Interaction>, With<SimPlayPauseButton>, With<Button>),
    >,
    mut sim_state: ResMut<SimulationState>,
) {
    for interaction in &mut interactions {
        if *interaction == Interaction::Pressed {
            sim_state.running = !sim_state.running;
        }
    }
}

pub(super) fn handle_sim_step_button_interaction(
    mut interactions: Query<
        &Interaction,
        (Changed<Interaction>, With<SimStepButton>, With<Button>),
    >,
    mut sim_state: ResMut<SimulationState>,
) {
    for interaction in &mut interactions {
        if *interaction == Interaction::Pressed {
            sim_state.running = false;
            sim_state.step_once = true;
        }
    }
}

pub(super) fn handle_sim_parallel_button_interaction(
    mut interactions: Query<
        &Interaction,
        (Changed<Interaction>, With<SimParallelButton>, With<Button>),
    >,
    mut parallel_settings: ResMut<SimulationParallelSettings>,
) {
    for interaction in &mut interactions {
        if *interaction == Interaction::Pressed {
            parallel_settings.enabled = !parallel_settings.enabled;
        }
    }
}

pub(super) fn handle_save_load_name_input_button_interaction(
    mut interactions: Query<
        &Interaction,
        (Changed<Interaction>, With<SaveLoadDialogNameInputButton>),
    >,
    mut save_load_ui_state: ResMut<SaveLoadUiState>,
) {
    if save_load_ui_state.mode != Some(SaveLoadDialogMode::Save) {
        return;
    }
    for interaction in &mut interactions {
        if *interaction == Interaction::Pressed {
            save_load_ui_state.input_focused = true;
        }
    }
}

pub(super) fn handle_save_load_dialog_buttons(
    mut confirm_buttons: Query<
        &Interaction,
        (Changed<Interaction>, With<SaveLoadDialogConfirmButton>),
    >,
    mut cancel_buttons: Query<
        &Interaction,
        (Changed<Interaction>, With<SaveLoadDialogCancelButton>),
    >,
    mut save_load_ui_state: ResMut<SaveLoadUiState>,
    mut save_writer: MessageWriter<SaveMapRequest>,
    mut load_writer: MessageWriter<LoadMapRequest>,
    mut replay_load_writer: MessageWriter<ReplayLoadScenarioRequest>,
    mut load_default_world_writer: MessageWriter<LoadDefaultWorldRequest>,
) {
    for interaction in &mut cancel_buttons {
        if *interaction == Interaction::Pressed {
            close_save_load_dialog(&mut save_load_ui_state);
            return;
        }
    }

    for interaction in &mut confirm_buttons {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match save_load_ui_state.mode {
            Some(SaveLoadDialogMode::Save) => {
                let slot_name = resolve_save_slot_name(&save_load_ui_state);
                let Some(slot_name) = slot_name else {
                    save_load_ui_state.status_message =
                        "Enter a save name or choose an existing slot".to_string();
                    return;
                };
                save_writer.write(SaveMapRequest { slot_name });
                close_save_load_dialog(&mut save_load_ui_state);
            }
            Some(SaveLoadDialogMode::Load) => {
                let Some(slot_name) = save_load_ui_state.selected_slot.clone() else {
                    save_load_ui_state.status_message = "Select a save slot to load".to_string();
                    return;
                };
                match save_load_ui_state.selected_slot_source {
                    Some(SaveLoadSlotSource::DefaultWorld) => {
                        load_default_world_writer.write(LoadDefaultWorldRequest);
                    }
                    Some(SaveLoadSlotSource::Save) | None => {
                        load_writer.write(LoadMapRequest { slot_name });
                    }
                    Some(SaveLoadSlotSource::TestCase) => {
                        replay_load_writer.write(ReplayLoadScenarioRequest {
                            scenario_name: slot_name,
                        });
                    }
                }
                close_save_load_dialog(&mut save_load_ui_state);
            }
            None => {}
        }
    }
}

pub(super) fn handle_save_load_slot_button_interaction(
    mut interactions: Query<
        (&Interaction, &SaveLoadSlotButton),
        (Changed<Interaction>, With<Button>),
    >,
    mut save_load_ui_state: ResMut<SaveLoadUiState>,
) {
    for (interaction, button) in &mut interactions {
        if *interaction != Interaction::Pressed {
            continue;
        }
        save_load_ui_state.selected_slot = Some(button.slot_name.clone());
        save_load_ui_state.selected_slot_source = Some(button.source);
        if matches!(save_load_ui_state.mode, Some(SaveLoadDialogMode::Save)) {
            if matches!(button.source, SaveLoadSlotSource::Save) {
                save_load_ui_state.input_name = button.slot_name.clone();
            }
            save_load_ui_state.input_focused = false;
            save_load_ui_state.ime_preedit.clear();
        } else {
            save_load_ui_state.input_focused = false;
        }
        save_load_ui_state.status_message.clear();
    }
}

pub(super) fn handle_save_load_text_input(
    mut keyboard_input_reader: MessageReader<KeyboardInput>,
    mut ime_reader: MessageReader<Ime>,
    mut save_load_ui_state: ResMut<SaveLoadUiState>,
) {
    let dialog_open = save_load_ui_state.mode.is_some();
    let save_mode_focused = save_load_ui_state.mode == Some(SaveLoadDialogMode::Save)
        && save_load_ui_state.input_focused;

    for ime in ime_reader.read() {
        if !save_mode_focused {
            continue;
        }
        match ime {
            Ime::Preedit { value, cursor, .. } if cursor.is_some() => {
                save_load_ui_state.ime_preedit = value.clone();
            }
            Ime::Preedit { cursor, .. } if cursor.is_none() => {
                save_load_ui_state.ime_preedit.clear();
            }
            Ime::Commit { value, .. } => {
                append_printable_text(&mut save_load_ui_state.input_name, value);
                save_load_ui_state.ime_preedit.clear();
            }
            _ => {}
        }
    }

    for keyboard_input in keyboard_input_reader.read() {
        if !keyboard_input.state.is_pressed() {
            continue;
        }
        if !dialog_open {
            continue;
        }
        if matches!(keyboard_input.logical_key, Key::Escape) {
            close_save_load_dialog(&mut save_load_ui_state);
            return;
        }
        if !save_mode_focused {
            continue;
        }
        match (&keyboard_input.logical_key, &keyboard_input.text) {
            (Key::Backspace, _) => {
                save_load_ui_state.input_name.pop();
                save_load_ui_state.ime_preedit.clear();
            }
            (_, Some(inserted_text)) => {
                append_printable_text(&mut save_load_ui_state.input_name, inserted_text);
            }
            _ => {}
        }
    }
}

pub(super) fn sync_save_load_ime_state(
    save_load_ui_state: Res<SaveLoadUiState>,
    mut primary_window: Single<&mut Window, With<PrimaryWindow>>,
) {
    let enable_ime = save_load_ui_state.mode == Some(SaveLoadDialogMode::Save)
        && save_load_ui_state.input_focused;
    primary_window.ime_enabled = enable_ime;
    if enable_ime {
        if let Some(cursor) = primary_window.cursor_position() {
            primary_window.ime_position = cursor;
        }
    }
}
