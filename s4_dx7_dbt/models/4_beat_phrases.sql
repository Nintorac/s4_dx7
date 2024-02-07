SELECT
	floor(e.start_time/4) bucket
	, track_id
	, concat(track_id, bucket::varchar) phrase_id
	, e.start_time-floor(e.start_time/4)*4 start_time
	, e.note
	, e.duration
	, e.velocity
FROM {{ source('midi', 'detailed_notes') }} e
where midi_id like 'f%'
order by e.track_id, e.start_time, e.note, e.duration, e.velocity asc
