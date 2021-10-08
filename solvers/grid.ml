open Core

open Client
open Timeout
open Task
open Utils
open Program
open Type

exception Exception of string


type dir_type =
{
	mutable delta_x : int;
	mutable delta_y : int;
}
let up = {delta_x = -1; delta_y = 0};;
let down = {delta_x = 1; delta_y = 0};;
let left = {delta_x = 0; delta_y = -1};;
let right = {delta_x = 0; delta_y = 1};;

type grid_state =
{
	mutable x : int;
	mutable y : int;
	mutable dir : dir_type;
	mutable pendown : bool;
	w : int;
	h : int;
	mutable board : (bool array) array;
};;

type grid_cont = grid_state -> grid_state ;;
let tgrid_cont = make_ground "grid_cont";;

let move_forward game =
	game.x <- max (min (game.x + game.dir.delta_x) (game.w-1)) 0;
	game.y <- max (min (game.y + game.dir.delta_y) (game.h-1)) 0;
	if game.pendown then game.board.(game.x).(game.y) <- true;;

let turn_left game =
	let rec rotate_left = function
		|{delta_x = -1; delta_y = 0} -> {delta_x = 0; delta_y = -1}
		|{delta_x = 1; delta_y = 0} -> {delta_x = 0; delta_y = 1}
		|{delta_x = 0; delta_y = -1} -> {delta_x = 1; delta_y = 0}
		|{delta_x = 0; delta_y = 1} -> {delta_x = -1; delta_y = 0}
		|{delta_x = x; delta_y = y} -> raise (Exception "Direction not handled")
	in
	game.dir <- (rotate_left game.dir);;

let turn_right game =
	let rec rotate_right = function
		|{delta_x = -1; delta_y = 0} -> {delta_x = 0; delta_y = 1}
		|{delta_x = 1; delta_y = 0} -> {delta_x = 0; delta_y = -1}
		|{delta_x = 0; delta_y = -1} -> {delta_x = -1; delta_y = 0}
		|{delta_x = 0; delta_y = 1} -> {delta_x = 1; delta_y = 0}
		|{delta_x = x; delta_y = y} -> raise (Exception "Direction not handled")
	in
	game.dir <- (rotate_right game.dir);;


ignore(primitive "grid_left" (tgrid_cont @> tgrid_cont)
	(fun (k: grid_cont) (s: grid_state) : grid_state ->
		turn_left(s);
		k(s)));;
ignore(primitive "grid_right" (tgrid_cont @> tgrid_cont)
	(fun (k: grid_cont) (s: grid_state) : grid_state ->
		turn_right(s);
		k(s)));;
ignore(primitive "grid_move" (tgrid_cont @> tgrid_cont)
	(fun (k: grid_cont) (s: grid_state) : grid_state ->
		move_forward(s);
		k(s)));;
ignore(primitive "grid_dopendown" (tgrid_cont @> tgrid_cont)
	(fun (k: grid_cont) (s: grid_state) : grid_state ->
		s.pendown <- true;
		k(s)));;
ignore(primitive "grid_dopenup" (tgrid_cont @> tgrid_cont)
	(fun (k: grid_cont) (s: grid_state) : grid_state ->
		s.pendown <- false;
		k(s)));;

let print_row my_array=
	Printf.eprintf "[|";
	for i = 0 to ((Array.length my_array)-1) do
	   Printf.eprintf "%b" my_array.(i);
	done;
	Printf.eprintf "|]";;
let print_matrix the_matrix =
	Printf.eprintf "[|\n";
	for i = 0 to ((Array.length the_matrix)-1) do
		if not (phys_equal i 0) then Printf.eprintf "\n" else ();
		print_row the_matrix.(i);
	done;
	Printf.eprintf "|]\n";;

ignore(primitive "grid_embed" ((tgrid_cont @> tgrid_cont) @> tgrid_cont @> tgrid_cont)
	(fun (body: grid_cont -> grid_cont) (k: grid_cont) (s: grid_state) : grid_state ->
		let x = s.x in
		let y = s.y in
		let pendown = s.pendown in
		let dir = s.dir in
		let _ = body (fun s -> s) s in
		s.x <- x;
		s.y <- y;
		s.pendown <- pendown;
		s.dir <- dir;
		let ns = k(s) in
		ns));;

let evaluate_GRID timeout p start x y =
    begin
      (* Printf.eprintf "%s\n" (string_of_program p); *)
      let p = analyze_lazy_evaluation p in
      let new_discrete =
        try
          match run_for_interval
                  timeout
                  (fun () -> run_lazy_analyzed_with_arguments p [fun s -> s] {board=start; w=(Array.length start); h=(Array.length start.(0)); dir=up; pendown=true; x=x; y=y})
          with
          | Some(p) ->
            Some(p.board)
          | _ -> None
        with | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
             (* we have to be a bit careful with exceptions *)
             (* if the synthesized program generated an exception, then we just terminate w/ false *)
             (* but if the enumeration timeout was triggered during program evaluation, we need to pass the exception on *)
             | otherException -> begin
                 if otherException = EnumerationTimeout then raise EnumerationTimeout else None
               end
      in
      new_discrete
    end
;;

register_special_task "GridTask" (fun extra ?timeout:(timeout = 0.001)
    name task_type examples ->
  assert (task_type = tgrid_cont @> tgrid_cont);
  assert (examples = []);

  let open Yojson.Basic.Util in
  let start : (bool array) array = extra |> member "start" |> to_list |>
		List.map ~f:(fun el -> el |> to_list |> List.map ~f:(fun el -> el |> to_bool) |> Array.of_list) |> Array.of_list
  in
  let goal : (bool array) array = extra |> member "goal" |> to_list |>
		List.map ~f:(fun el -> el |> to_list |> List.map ~f:(fun el -> el |> to_bool) |> Array.of_list) |> Array.of_list
	in
	let x = extra |> member "location" |> index 0 |> to_int
	in
	let y = extra |> member "location" |> index 1 |> to_int
	in

  (* Printf.eprintf "TARGETING:\n%s\n\n" *)

  { name = name    ;
    task_type = task_type ;
    log_likelihood =
      (fun p ->
				 let s : (bool array) array = Array.map ~f:(Array.copy) start in
         let hit = (evaluate_GRID timeout p s x y = Some(goal)) in
         (*Printf.eprintf "\t%s %b\n\n" (string_of_program p) hit;*)
         if hit then 0. else log 0.)
  })
;;
